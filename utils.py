import time
from bs4 import BeautifulSoup, Comment, Tag
import re
import unicodedata
from typing import List, Optional
import json
import dotenv
from groq import Groq
import requests
from sklearn.metrics.pairwise import cosine_similarity as similarity
import numpy as np
import networkx as nx
from neo4j import GraphDatabase


class BlockNode:
    def __init__(self, tag, attributes=None, content=None):
        self.tag = tag
        self.path = []
        self.attributes = attributes or {}
        self.content = content.strip() if content else ""
        self.children = []
        self.block = ""
        self.is_leaf = False
        self.embedding = None

    def add_child(self, child):
        self.children.append(child)

    def to_dict(self):
        return {
            "tag": self.tag,
            "attributes": self.attributes,
            "content": self.content,
            "children": [child.to_dict() for child in self.children],
        }

    def __str__(self):
        return f"Tag: {self.tag}, Content: {self.content}, children: {len(self.children)}, tag: {self.tag}"


def clean_html(html_content: str) -> BeautifulSoup:
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove <script>, <style>, and comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Clean attributes (remove inline styles, overly long values)
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr == "style" or len(str(tag[attr])) > 50:
                del tag[attr]

    # Merge single-nested identical tags
    def merge_single_nested_tags(tag):
        while (
            len(tag.contents) == 1
            and isinstance(tag.contents[0], Tag)
            and tag.contents[0].name == tag.name
        ):
            child = tag.contents[0]
            for k, v in child.attrs.items():
                if k not in tag.attrs:
                    tag.attrs[k] = v
            child.unwrap()
        for child in tag.find_all(recursive=False):
            merge_single_nested_tags(child)

    for tag in soup.find_all(True):
        merge_single_nested_tags(tag)

    # Normalize and clean text content
    def normalize_text(text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"xa0\d*", " ", text)
        text = re.sub(r"\s+", " ", text)
        return unicodedata.normalize("NFKC", text).strip()

    for text_node in soup.find_all(string=True):
        if isinstance(text_node, str):
            cleaned = normalize_text(text_node)
            if cleaned:
                text_node.replace_with(cleaned)
            else:
                text_node.extract()

    # Remove empty tags
    def remove_empty_tags(tag):
        for child in tag.find_all(recursive=False):
            remove_empty_tags(child)
        if not tag.contents or all(
            str(c).strip() in ["", "None"] for c in tag.contents
        ):
            tag.decompose()

    remove_empty_tags(soup)

    return soup


def build_dom_tree(html_content):
    def build_tree(element):
        if not getattr(element, "name", None):
            return None

        # FIX: Collect only direct text children (same as original)
        direct_text = []
        for child in element.children:
            if isinstance(child, str):
                text = child.strip()
                if text:
                    direct_text.append(text)

        # ✨ Construct node first
        node = BlockNode(
            tag=element.name, attributes=element.attrs, content=" ".join(direct_text)
        )

        # Recursively build child nodes
        for child in element.children:
            child_node = build_tree(child)
            if child_node:
                node.add_child(child_node)

        return node

    def losen_structure(node):
        if not node:
            return
        for child in node.children:
            losen_structure(child)
        i = 0
        while i < len(node.children):
            child = node.children[i]
            if not child.content.strip() and len(child.children) == 1:
                node.children[i] = child.children[0]
            else:
                i += 1

    root = build_tree(html_content)
    losen_structure(root)
    return root


def build_block_tree(dom_tree: BlockNode, max_window: int):
    def safe_token_count(text):
        return len((text or "").split())

    def merge_upward(node, limit):
        if not node:
            return True

        # Initialize block from content or empty string
        node.block = node.content or ""

        # Leaf node
        if not node.children:
            node.is_leaf = True
            return True

        # Recursively process all children
        child_is_leaf = []
        for child in node.children:
            child_is_leaf.append(merge_upward(child, limit))

        all_leaf = all(child_is_leaf)

        # Flatten linear branches (one-child node)
        if not all_leaf and len(node.children) == 1:
            child = node.children[0]
            child_block = child.block or child.content or ""
            node.block = (node.content or "") + " " + child_block
            node.children = child.children
            return False

        # Try to merge children if they are all leaves and token count is under limit
        if all_leaf:
            merged_parts = []
            for child in node.children:
                part = child.block or child.content or ""
                part = part.strip()
                if part:
                    merged_parts.append(part)

            token_total = safe_token_count(node.content) + sum(
                safe_token_count(p) for p in merged_parts
            )

            if token_total <= limit:
                node.block = (node.content or "").strip()
                if node.block and merged_parts:
                    node.block += " " + " ".join(merged_parts)
                elif not node.block:
                    node.block = " ".join(merged_parts)
                node.children = []
                node.is_leaf = True
                return True

        # Cannot merge: retain structure
        node.block = node.block or node.content or ""
        node.is_leaf = False
        return False

    def retag_tree(node, tag_counts=None):
        tag_counts = tag_counts or {}
        prefix = "".join(filter(str.isalpha, node.tag)) or "node"
        tag_counts[prefix] = tag_counts.get(prefix, 0) + 1
        node.tag = f"{prefix}{tag_counts[prefix]}"
        node.path = node.path + [node.tag]
        for child in node.children:
            child.path = node.path.copy()
            retag_tree(child, tag_counts)

    # Normalize root tag
    if dom_tree.tag == "[document]":
        dom_tree.tag = "document"

    retag_tree(dom_tree)
    merge_upward(dom_tree, max_window)
    return dom_tree


def rephrase_question(question: str):
    client = Groq(api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"))

    try:
        prompt_system = {
            "role": "system",
            "content": (
                "You are a structured rephrasing assistant. Your task is to generate exactly *two* diverse but semantically equivalent "
                "paraphrases of a factual question, for use in embedding similarity evaluation.\n\n"
                "Output Requirements:\n"
                "- Return only a single valid JSON object with *three* keys: `paraphrase_1`, `paraphrase_2`.\n"
                "- Do *NOT* include extra text, explanations, or markdown formatting (do not use ```json).\n"
                "- All values must be in valid JSON strings.\n\n"
                "Strict Output Format:\n"
                "{\n"
                '  "paraphrase_1": "string",\n'
                '  "paraphrase_2": "string"\n'
                "}\n\n"
                "Guidelines:\n"
                "- The paraphrases must preserve the meaning of the original factual question.\n"
                "- `paraphrase_1` should be a mild rewording.\n"
                "- `paraphrase_2` should use significantly different phrasing or syntax (e.g., question form, passive voice, inversion).\n"
                "- Only valid JSON output is allowed. Any malformed JSON output is considered a failure.\n"
            ),
        }

        prompt_user = {
            "role": "user",
            "content": f'Paraphrase question:\n"{question}"',
        }
        few_shot_examples = [
            {
                "role": "user",
                "content": 'Paraphrase question:\n"What year did the Berlin Wall fall?"',
            },
            {
                "role": "assistant",
                "content": """
                    {"paraphrase_1": "In which year was the Berlin Wall taken down?","paraphrase_2": "When did the Berlin Wall come down?"}
                """,
            },
            {
                "role": "user",
                "content": 'Paraphrase question:\n"Who discovered penicillin?"',
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "paraphrase_1": "Which scientist is credited with the discovery of penicillin?",
                        "paraphrase_2": "Who is known for discovering penicillin?",
                    }
                ),
            },
        ]
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[prompt_system] + few_shot_examples + [prompt_user],
            temperature=0.5,
            response_format={
                "type": "json_object",
                "json_object": {
                    "paraphrase_1": "string",
                    "paraphrase_2": "string",
                    "strict": ["paraphrase_1", "paraphrase_2"],
                },
            },
            stream=False,
        )

        result = response.choices[0].message.content
        parsed = json.loads(result)
        print(parsed)
        output = [question, parsed["paraphrase_1"], parsed["paraphrase_2"]]
        print(output)
        return output

    except Exception as e:
        print(f"Error during question paraphrasing: {e}")
        return [question]


def fetch_embedding(text):
    EMB_MODEL_ID = "text-embedding-mxbai-embed-large-v1"
    """Fetch embedding for a given text from LMStudio."""
    try:
        url = "http://localhost:9999/v1/embeddings"
        data = {"model": EMB_MODEL_ID, "input": text}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # print(response)
        if response.status_code == 200:
            # print(response.json())
            return response.json()["data"]
        else:
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error querying LMStudio: {e}")
        return None


def tree_contents(root):
    contents = []
    if not root:
        return contents
    if root.block.strip():
        contents.append(root.block)
    for child in root.children:
        child_content = tree_contents(child)
        child_content = [content for content in child_content if content.strip()]
        contents.extend(child_content)
    return contents


def embed_contents(root: BlockNode, contents):
    batch_size = 200
    all_embeddings = []

    for i in range(0, len(contents), batch_size):
        contents_batch = contents[i : i + batch_size]
        embd = fetch_embedding(contents_batch)
        if embd is None:
            raise RuntimeError(f"Embedding failed for batch {i}-{i + batch_size}")
        all_embeddings.extend(embd)

    def tranverse_tree(root: BlockNode, idx):
        # print(root)
        if not root:
            return
        if root.block.strip():
            root.embedding = all_embeddings[idx]["embedding"]
            # print(root.content, idx, root.embedding)
            idx += 1
        for child in root.children:
            idx = tranverse_tree(child, idx)
            # print(idx)
        return idx

    expected = len([c for c in contents if c.strip()])
    if len(all_embeddings) != expected:
        raise ValueError(f"Expected {expected} embeddings, got {len(all_embeddings)}")
    tranverse_tree(root, 0)


def gather_similarity_scores(node: BlockNode, ques_embds, scores=None):
    if scores is None:
        scores = []

    if node is None:
        return scores

    if node.embedding is not None:
        sims = [similarity([node.embedding], [q_embd])[0][0] for q_embd in ques_embds]
        aggregated_score = max(sims)  # Use max similarity for the node
        cosine_sim = aggregated_score
        scores.append(cosine_sim)

    for child in node.children:
        gather_similarity_scores(child, ques_embds, scores)

    return scores


def compute_similarity_threshold(scores, min_keep=40, top_percent=0.3):
    if not scores:
        return 0.1  # fallback for empty input

    scores = np.array(scores)
    num_scores = len(scores)

    # Sort scores descendingly
    sorted_scores = np.sort(scores)[::-1]

    # Determine the index that corresponds to the top_percent
    top_k = max(int(top_percent * num_scores), min_keep)

    # Enforce at least min_keep elements
    top_k = min(len(sorted_scores), max(min_keep, int(top_percent * num_scores)))

    # Select the threshold as the k-th highest score
    threshold = sorted_scores[top_k - 1]

    return threshold


def similarity_compute(node: BlockNode, ques_embd, total_sim, max_sim, min_sim):
    if not node:
        return
    if node.embedding is not None:
        cosine_sim = similarity([node.embedding], [ques_embd])[0][0]
        # print(cosine_sim, node.block) if cosine_sim > 0.5 else None
        max_sim = max(max_sim, cosine_sim)
        min_sim = min(min_sim, cosine_sim)
        total_sim += cosine_sim

    for child in node.children:
        max_sim, min_sim, total_sim = similarity_compute(
            child, ques_embd, total_sim, max_sim, min_sim
        )
    return max_sim, min_sim, total_sim


def pruning_tree_on_similarity(node: BlockNode, ques_embd, threshold):
    if not node:
        return None
    if node.embedding is not None:
        cosine_sim = similarity([node.embedding], [ques_embd])[0][0]

        pruned_children = []

        for child in node.children:
            pruned_child = pruning_tree_on_similarity(child, ques_embd, threshold)
            if pruned_child is not None:
                pruned_children.extend(pruned_child)

        if cosine_sim < threshold:
            return pruned_children if pruned_children else None

        node.children = pruned_children
        return [node]
    return node.children


def get_node_paths_and_contents(node: BlockNode, path="") -> list:
    """Collect paths and block content from all nodes in the tree."""
    if not node:
        return []

    current_path = f"<{node.path[-1]}>"
    results = []

    if node.block.strip():  # Only include non-empty blocks
        results.append({"path": current_path, "content": node.block})

    for child in node.children:
        results.extend(get_node_paths_and_contents(child, current_path))

    return results


def inference(html_content, question):
    client = Groq(
        # This is the default and can be omitted
        api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"),
    )
    print(html_content)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                You are an expert system designed to evaluate and prune irrelevant or redundant HTML content. Your task is to process an HTML document and retain only the text blocks that are directly or indirectly relevant to a given question.
                Text is considered relevant if it:\n
                    1. Provides a direct answer to the question.\n
                    2. Supplies evidence or details that support answering the question.\n
                    3. Offers necessary context for answering the question.\n
                Avoid retaining blocks that:\n
                    1. Are unrelated to the question.\n
                    2. Contain redundant or repetitive information.\n
                    3. Discuss general information not connected to the topic.\n
                You should output only a path to the relevant blocks, preserving the original structure.
                **Input**:
                **HTML List**: {List of HTML tags and contents}.
                **Question**: {Question}.
                **Output**: A JSON object containing a list of text blocks' paths and their relevant text.

                **Output Format** Must follow this JSON format do not use any special character such as '\n' for newline or similar :
                {
                    "relevant_blocks": [
                    {
                        "path": "<html1><title1>",
                        "text": "Relevant Title"
                    },
                    {
                        "path": "<body1><p3>",
                        "text": "Supported Information"
                    },
                    {
                        "path: "<h100>",
                        "text": "Sufficient Information"
                    }
                    ]
                }
                """,
                },
                {
                    "role": "user",
                    "content": """
**HTML List**:
[
  "<tr1>Albert Einstein was awarded the Nobel Prize in Physics in 1921.",
  "<tr2>Photosynthesis is the process used by plants to convert sunlight into chemical energy.",
  "<tr3>Einstein's work on the photoelectric effect helped establish quantum theory."
]

**Question**: "What scientific contributions is Albert Einstein known for?"
Please output in JSON format. {"relevant_blocks": []}
            """,
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "relevant_blocks": [
                                {
                                    "path": "<tr1>",
                                    "text": "Albert Einstein was awarded the Nobel Prize in Physics in 1921.",
                                },
                                {
                                    "path": "<tr3>",
                                    "text": "Einstein's work on the photoelectric effect helped establish quantum theory.",
                                },
                            ]
                        }
                    ),
                },
                {
                    "role": "user",
                    "content": f'**HTML Lists**: "{html_content}"\n**Question**: "{question}"\nPlease output in JSON format. {{"relevant_blocks": []}}',
                },
            ],
            response_format={
                "type": "json_object",  # Use 'json_object' as the allowed type
                "json_object": {
                    "relevant_blocks": [
                        {
                            "path": "string",  # Define the expected structure of the JSON object,
                            "text": "string",
                        }
                    ],
                    "strict": ["relevant_blocks"],
                },
            },
            # model="llama3-70b-8192",
            # model="meta-llama/llama-prompt-guard-2-22m",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            stream=False,
            max_tokens=1000,
        )

    except Exception as e:
        print(f"Error: {e}")
        return None
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def prepare_data_for_llm(tree: BlockNode, chunk_size: int = 10000):
    """Divide the flattened path-content pairs into chunks respecting chunk size."""
    node_items = get_node_paths_and_contents(tree)

    chunks = []
    current_chunk = []
    current_length = 0

    for item in node_items:
        item_str = item["path"] + item["content"]
        item_len = len(item_str)

        if current_length + item_len > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [item_str]
            current_length = item_len
        else:
            current_chunk.append(item_str)
            current_length += item_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def reformat_llm_output(tree: BlockNode, chunks: list, question: str):
    """Send each chunk to LLM, re-assemble the relevant blocks, and map them back to full paths."""
    raw_responses = []

    for chunk in chunks:
        response = inference(chunk, question)
        if response:
            raw_responses.append(response)
        time.sleep(5)

    # Combine all relevant block paths
    final_block_refs = []
    for response in raw_responses:
        final_block_refs.extend(json.loads(response)["relevant_blocks"])

    # Clean up malformed paths and ensure 'text' field is always present
    for block in final_block_refs:
        path_parts = block["path"].split(">")
        miss_parse_count = 0

        if len(path_parts) > 1:
            for part in path_parts[:-1]:
                if not part.startswith("<"):
                    if "text" not in block:
                        block["text"] = ""
                    if part not in block["text"]:
                        block["text"] += part
                else:
                    miss_parse_count += 1

            last_part = path_parts[-1]
            if "text" not in block:
                block["text"] = ""
            if last_part not in block["text"]:
                block["text"] += " " + last_part

            # Fix path to last valid segment
            if miss_parse_count:
                block["path"] = path_parts[miss_parse_count - 1] + ">"

    # Match cleaned paths to actual tree nodes
    matched_blocks = []

    def traverse_and_match(node: BlockNode):
        if not node:
            return

        current_path = f"<{node.path[-1]}>"
        for block in final_block_refs:
            if block["path"] == current_path:
                matched_blocks.append(
                    {
                        "path": node.path,
                        "text": block.get("text", ""),
                        "content": node.block,
                    }
                )

        for child in node.children:
            traverse_and_match(child)

    traverse_and_match(tree)
    return matched_blocks


def validate_final_tree_contents(G, short_answer_list):
    client = Groq(api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"))

    try:
        # Extract and format content from graph nodes
        node_texts = [
            {"id": str(node), "text": str(G.nodes[node]["content"])} for node in G.nodes
        ]
        all_node_texts = "\n".join([f"{n['text']}" for n in node_texts])

        short_answers = [" ".join(answer) for answer in short_answer_list]
        #         f"""
        # You are evaluating how well a list of short answers is covered by content from an HTML document.\n

        # Each short answer is matched to segments of text (with node IDs) extracted from the HTML. Evaluate the degree of representation for each short answer based on the content.\n

        # **Scoring Criteria:**
        # 1.0 → Clearly represented or strongly paraphrased\n
        # 0.5 → Loosely represented or partially present\n
        # 0.0 → Not represented or unrelated\n

        # For each short answer, return:
        # - the short answer string\n
        # - a score from [1.0, 0.5, 0.0]\n
        # - a list of the matching node contents\n

        prompt_user = {
            "role": "user",
            "content": (
                "Evaluate representation:\n\n"
                f"**Node Texts:**\n{chr(10).join(all_node_texts)}\n\n"
                f"**Short Answers**:\n{json.dumps(short_answers, indent=2)}"
            ),
        }

        prompt_system = {
            "role": "system",
            "content": (
                "You are a model that evaluates whether a list of short factual answers is represented in text segments from HTML nodes.\n\n"
                "Scoring Criteria:\n"
                "1.0 → Clearly represented or strongly paraphrased.\n"
                "0.0 → Not represented or unrelated.\n\n"
                "Output must be a valid JSON object in the following format:\n"
                "{\n"
                '  "representation": [\n'
                "    {\n"
                '      "answer": "string",\n'
                '      "score": 1.0 | 0.0,\n'
                '      "source_node_contents": ["string"]\n'
                "    }"
                "  ]"
                "}\n\n"
                "Do not include any explanation, comments, or markdown. Respond with only a JSON object."
            ),
        }

        few_shot = [
            {
                "role": "user",
                "content": (
                    "Evaluate representation:\n\n"
                    "**Node Texts:**\n"
                    "a12 Isaac Newton laid the groundwork for classical mechanics.\n"
                    "a3 Marie Curie was a pioneer in radioactivity research.\n"
                    " Albert Einstein developed   the theory of relativity.\n"
                    "Niels Bohr contributed to quantum theory.\n"
                    "\n**Short Answers**:\n"
                    '["Albert Einstein", "Marie Curie"]'
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "representation": [
                            {
                                "answer": "Albert Einstein",
                                "score": 1.0,
                                "source_node_contents": [
                                    "Albert Einstein developed the theory of relativity."
                                ],
                            },
                            {
                                "answer": "Marie Curie",
                                "score": 1.0,
                                "source_node_contents": [
                                    "Marie Curie was a pioneer in radioactivity research."
                                ],
                            },
                        ]
                    },
                    indent=2,
                ),
            },
            # {
            #     "role": "user",
            #     "content": (
            #         "Evaluate representation:\n\n"
            #         "**Node Texts:**\n"
            #         "Photosynthesis is the process by which green plants convert sunlight into energy using chlorophyll.\n"
            #         "\n**Short Answers**:\n"
            #         '["Photosynthesis"]'
            #     ),
            # },
            # {
            #     "role": "assistant",
            #     "content": json.dumps(
            #         {
            #             "representation": [
            #                 {
            #                     "answer": "Photosynthesis",
            #                     "score": 1.0,
            #                     "source_node_contents": [
            #                         "Photosynthesis is the process by which green plants convert sunlight into energy using chlorophyll."
            #                     ],
            #                 },
            #             ]
            #         },
            #         indent=2,
            #     ),
            # },
        ]
        response = client.chat.completions.create(
            # model="meta-llama/llama-4-maverick-17b-128e-instruct",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[prompt_system] + few_shot + [prompt_user],
            temperature=0.1,
            response_format={
                "type": "json_object",  # Use 'json_object' as the allowed type
                "json_object": {
                    "representation": [
                        {
                            "answer": "string",
                            "score": "number",
                            "source_node_contents": ["string"],
                        }
                    ],
                    "strict": ["representation"],
                },
            },
        )

        result = response.choices[0].message.content
        print(result)
        print(all_node_texts)
        # 4. Parse and compute score
        parsed = json.loads(result)
        scores = [entry["score"] for entry in parsed["representation"]]
        total = sum(scores)
        coverage_ratio = total / len(scores) if scores else 0.0
        print(f"\nCoverage Score: {coverage_ratio:.2f}")
        return {"coverage_score": coverage_ratio, "details": parsed["representation"]}

    except Exception as e:
        print(f"Error during semantic coverage check: {e}")
        print(result)
        return None


def heuristic_function(path1, path2):
    common_prefix = 0
    for i, j in zip(path1, path2):
        if i == j:
            common_prefix += 1
        else:
            break
    depth1 = len(path1)
    depth2 = len(path2)
    depth_difference = abs(depth1 - depth2)
    # 3. Sibling relationship
    sibling_score = 0
    if depth1 == depth2 and path1[:-1] == path2[:-1]:
        sibling_score = 1

    # # 4. Custom tag weights
    # weight_score = 0
    # for tag in path1 + path2:
    #     weight_score += tag_weights.get(tag, 0)

    # Combine factors into a single heuristic score
    relationship_score = (
        common_prefix * 2  # Common prefix is more important
        - depth_difference * 0.5  # Penalize depth differences
        + sibling_score * 3  # Reward sibling relationship
        # + weight_score * 0.1  # Incorporate tag weights
    )

    return max(relationship_score, 0)


def build_graph(block_path, embd_score):
    G = nx.DiGraph()

    for idx, path in enumerate(block_path):
        # print(path['text'])
        G.add_node(
            " ".join(path["path"]),
            tag=path["path"][-1],
            content=path["content"],
            path=path["path"],
            embd=embd_score[idx]["embedding"],
        )

    nodes = list(G.nodes(data="embd"))
    # print(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # nodes[i][1] = [float(x) for x in nodes[i][1]]
            # nodes[j][1] = [float(x) for x in nodes[j][1]]
            structure_rel_score = heuristic_function(
                nodes[i][0].split(), nodes[j][0].split()
            )
            simi_score = similarity([nodes[i][1]], [nodes[j][1]])[0][0]
            G.add_edge(
                nodes[i][0],
                nodes[j][0],
                relationship="semantic",
                weight=simi_score * 0.5 + structure_rel_score * 0.5,
            )

    labels = {node: G.nodes[node]["content"] for node in G.nodes}

    # Remove bottom 20% of edges by weight
    all_edges = list(G.edges(data=True))
    all_weights = [edge[2]["weight"] for edge in all_edges]
    threshold = sorted(all_weights)[int(len(all_weights) * 0.5)]  # 50% quantile

    for u, v, data in all_edges:
        if data["weight"] <= threshold:
            G.remove_edge(u, v)

    # Draw with content as labels
    nx.draw(G, with_labels=True, labels=labels, pos=nx.spring_layout(G))
    # nx.draw(G, with_labels=True)
    return G


# Replace with your actual info
uri = "bolt://localhost:7687"
user = "neo4j"
password = "1234567890"


def upload_graph_to_neo4j(graph, driver):
    with driver.session() as session:
        # session.execute_write(_clear_existing_graph)
        for node, data in graph.nodes(data=True):
            session.execute_write(_create_node, node, data)
        for u, v, edge_data in graph.edges(data=True):
            session.execute_write(_create_edge, u, v, edge_data)


def _clear_existing_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def _create_node(tx, node_id, data):
    tx.run(
        """
        MERGE (n:Block {id: $id})
        SET n.content = $content,
            n.tag = $tag,
            n.path = $path
        """,
        id=node_id,
        content=data.get("content", ""),
        tag=data.get("tag", ""),
        path=" ".join(data.get("path", [])),
    )


def _create_edge(tx, from_node, to_node, data):
    tx.run(
        """
        MATCH (a:Block {id: $from_id}), (b:Block {id: $to_id})
        MERGE (a)-[r:SEMANTIC_RELATION]->(b)
        SET r.weight = $weight
        """,
        from_id=from_node,
        to_id=to_node,
        weight=data.get("weight", 0.0),
    )
