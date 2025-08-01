# %% [markdown]
# # Load Package

# %%
from datasets import load_dataset
import json
import requests
from bs4 import BeautifulSoup, Comment
from bs4 import Tag
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import networkx as nx
import numpy as np
import os
import time

similarity = cosine_similarity

# %% [markdown]
# # Load Dataset

# %%
ds = load_dataset(
    "google-research-datasets/natural_questions", "dev", split="validation"
)

# %%
for i in range(10):
    print(ds[i]["document"]["title"])
    print(ds[i]["document"]["url"])

# %% [markdown]
# # Immediate Run

# %% [markdown]
# ## Init Funcion

# %% [markdown]
# ### DOM Cleaning
# 

# %%
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
        return f'Tag: {self.tag}, Content: {self.content}, children: {len(self.children)}, tag: {self.tag}'
    
def clean_html(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Step 1: Remove <script>, <style>, and comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove attributes: (1) lengthy attributes (2) all `style` attributes
    def clean_attributes(tag):
        for attr in list(tag.attrs.keys()):
            if attr == "style" or len(str(tag[attr])) > 50:  # Remove inline styles and long attributes
                del tag[attr]

    for tag in soup.find_all(True):  # All tags
        clean_attributes(tag)

    # Step 2: Lossless Structural Compression
    # Merge single-nested tags safely
    def merge_single_nested_tags(tag):
        while (
            len(tag.contents) == 1
            and isinstance(tag.contents[0], Tag)
            and tag.contents[0].name == tag.name
        ):
            child = tag.contents[0]
            # Merge attributes safely
            for key, value in child.attrs.items():
                if key not in tag.attrs:
                    tag.attrs[key] = value
            child.unwrap()  # Unwrap instead of modifying contents directly

        for child in tag.find_all(recursive=False):  # Iterate safely
            merge_single_nested_tags(child)

    for tag in soup.find_all(True):  # All tags
        merge_single_nested_tags(tag)

    # Remove all \t and \n in the text content
    for text in soup.find_all(string=True):
        text.replace_with(text.replace("\n", "").replace("\t", ""))

    # Remove empty tags safely
    def remove_empty_tags(tag):
        for child in tag.find_all(recursive=False):
            remove_empty_tags(child)  # Recursively clean
        if not tag.contents or all(str(content).strip() in ["None",""]  for content in tag.contents):
            tag.decompose()

    remove_empty_tags(soup)

    return soup

def build_dom_tree(html_content):
    def build_tree(element):
        if not element.name:
            return None
        
        direct_text = []
        for child in element.children:
            if isinstance(child, str):
                text = child.strip()
                if text:
                    direct_text.append(text)

        # Recursively build child nodes
        children_nodes = []
        for child in element.children:
            if hasattr(child, 'name') or isinstance(child, str):
                child_node = build_tree(child)
                if child_node:
                    children_nodes.append(child_node)
        # If no direct text, use descendant content (especially for structural tags)
        node_content = (
            " ".join(direct_text)
        )

        node = BlockNode(
            tag=element.name,
            attributes=element.attrs,
            content=node_content.strip()
        )

        for child_node in children_nodes:
            node.add_child(child_node)

        return node

    def losen_structure(node: BlockNode):
        if not node:
            return

        for child in node.children:
            losen_structure(child)

        i = 0
        while i < len(node.children):
            child = node.children[i]
            if (
                not child.content.strip()
                and len(child.children) == 1
            ):
                del node.children[i]
                node.children.insert(i, child.children[0])
            else:
                i += 1

    root = build_tree(html_content)
    losen_structure(root)
    return root

def build_block_tree(dom_tree: BlockNode, max_window: int):
    """
    Build a bottom-up balanced block tree like HTML-RAG:
    - Merge sibling leaf nodes up to max_window tokens.
    - Merge linear branch segments.
    - Retag and track paths for traceability.
    """
    
    def count_tokens(node: BlockNode) -> int:
        text = node.block or node.content or ""
        return len(text.split())
    
    def flatten_text(node: BlockNode) -> str:
        if node.block:
            return node.block.strip()
        if not node.children:
            return (node.content or "").strip()
        parts = [(node.content or "").strip()] + [flatten_text(child) for child in node.children]
        return " ".join(p for p in parts if p)

    def merge_upward(node: BlockNode, limit: int):
        """Merge leaf siblings into a single block if within limit."""
        # go to sub-leaf nodes
        # check if leaf nodes can all merge, 
        # if can merge, merge them all, and go to ancestor.
            # check if the block created have rooms for ancestor.
            # if can merge, merge them all, and go to ancestor.
            # recursively

        # else, go to ancestor
            # try to merge intermediate nodes
        if not node:
            return True
        # print("Node content: ", node.content)

        node.block = node.content if node.content else ""
        if not node.children:
            # print("leaf node")
            # If no children, treat as leaf node
            node.is_leaf = True
            return node.is_leaf
        
        is_leaf = True
        for child in node.children:
            is_leaf = merge_upward(child, limit) and is_leaf
        # print("is child all leaf: ", is_leaf)
        if is_leaf is False:
            # check if the node have only one child
            # true, merge the child to node and assign the child children to node children
            # false, return False
            if len(node.children) == 1:
                child = node.children[0]
                node.block = (node.content if node.content else "") + (child.block if child.block else child.content)
                # print("node block: ", node.block)
                node.children = child.children
            return False
        else:
            # check if the children content and node content can be merged
            # true, merge the children content and self content to self block, return True
            # false, return False
            child_content = []
            for child in node.children:
                if not child.block:
                    if not child.content:
                        continue
                    else:
                        child_content.append(child.content)
                else:
                    child_content.append(child.block)
            token_count = len(node.content.split()) + sum(len(s.split()) for s in child_content)
            if token_count < limit:
                total_content = node.content + " " + " ".join(child_content)
                node.block = total_content
                # print("node block: ", node.block)
                # print("Total content: ", total_content)
                node.children = []
                node.is_leaf = True
                return True
            else:
                # print("token len: ", token_count)
                # If the content exceeds the limit, keep the children as is
                if not node.block:
                    node.block = node.content or ""
                # print("node block: ", node.block)
                # print(len(node.children))
                node.is_leaf = False
                return False
    def retag_tree(node: BlockNode, tag_counts=None):
        if tag_counts is None:
            tag_counts = {}

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
    # Optional: enable branch merging
    # merge_branch_segments(dom_tree, max_window)

    return dom_tree


# %% [markdown]
# ### Tree Pruning

# %%
EMB_MODEL_ID = "text-embedding-nomic-embed-text-v1.5-embedding"
# EMB_MODEL_ID = "text-embedding-mxbai-embed-large-v1"
def fetch_embedding(text):
    """Fetch embedding for a given text from LMStudio."""
    try:
        url = "http://localhost:9999/v1/embeddings"
        data = {"model": EMB_MODEL_ID, "input": text}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response)
        if response.status_code == 200:
            # print(response.json())
            return response.json()["data"]
        else:
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error querying LMStudio: {e}")
        return None
    # """Fetch embeddings from LMStudio for one or more texts."""
    # try:
    #     url = "http://localhost:9999/v1/embeddings"
    #     data = {"model": EMB_MODEL_ID, "input": text}
    #     headers = {"Content-Type": "application/json"}
    #     response = requests.post(url, headers=headers, data=json.dumps(data))
    #     if response.status_code == 200:
    #         response_json = response.json()
    #         if "data" in response_json:
    #             return response_json["data"]
    #         else:
    #             raise ValueError("Missing 'data' in response")
    #     else:
    #         print(f"HTTP {response.status_code}: {response.text}")
    #         return None
    # except requests.exceptions.RequestException as e:
    #     print(f"Error querying LMStudio: {e}")
    #     return None
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
        contents_batch = contents[i:i + batch_size]
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

def similarity_compute(node: BlockNode, ques_embd, total_sim, max_sim, min_sim):
    if not node:
        return
    if node.embedding is not None:
        cosine_sim = similarity([node.embedding], [ques_embd])[0][0]
        print(cosine_sim, node.block) if cosine_sim > 0.5 else None
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

# %%
def get_node_paths_and_contents(node: BlockNode, path="") -> list:
    """Return a list containing each node path and node block content."""
    if not node:
        return []
    # print(node)
    current_path = f"<{node.path[-1]}>"
        
    result = [{"path": current_path, "content": node.block}]
    
    if not node.block.strip():
        result.pop()
    for child in node.children:
        result.extend(get_node_paths_and_contents(child, current_path))
    return result

client = Groq(
    # This is the default and can be omitted
    api_key=os.getenv("GROQ_API"),
)


def inference(html_content, question):
    print(len(html_content))
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
                    "strict": ["relevant_blocks"]
                },
            },
            # model="llama-3.1-8b-instant",
            model="llama3-70b-8192",
            temperature=0.5,
            stream=False,
            max_tokens=2000,
        )

    except Exception as e:
        print(f"Error: {e}")
        return None
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

# %%
def prepare_data_for_llm(tree: BlockNode):
    chunk_size = 1000
    node_paths_and_contents = get_node_paths_and_contents(tree)
    # print(node_paths_and_contents)
    node_paths_and_contents_zip = [[node_paths_and_contents[0]['path']+node_paths_and_contents[0]['content']]]
    for path in node_paths_and_contents[1:]:
        if len(path["path"]) + len(path["content"]) < chunk_size - len(''.join(node_paths_and_contents_zip[-1])):
            node_paths_and_contents_zip[-1].append(path["path"]+path["content"])
        else:
            node_paths_and_contents_zip.append([path["path"]+path["content"]])
    return node_paths_and_contents_zip

# %% [markdown]
# ### Aggregate the pruning

# %%
def reformat_llm_output(tree, node_paths_and_contents_zip: list, question):
    raw_answer = []
    for path in node_paths_and_contents_zip:
        
        path_response = inference(path, question)
        
        if path_response:
            raw_answer.append(path_response)

        time.sleep(5)

    final_path_list = []
    for path in raw_answer:
        final_path_list.extend(json.loads(path)["relevant_blocks"])
    for path in final_path_list:
        print(path)
    
    for path in final_path_list:
        element = path["path"].split(">")
        miss_parse = 0
        if len(element) > 1:
            for elem in element[:-1]:
                if not elem.startswith("<"):
                    print(path["text"], elem)
                    path["text"] = path["text"] + elem if elem not in path["text"] else ""
                else:
                    miss_parse += 1
            # print(element, miss_parse)
            if "text" not in path:
                path["text"] = ""
            path["text"] += (" " + element[-1]) if element[-1] not in path["text"] else ""
            path["path"] = element[miss_parse - 1] + ">"
        print(path)

    block_path = []
    def get_full_path(node: BlockNode):
        if not node:
            return
        last_path = f"<{node.path[-1]}>"
        # print(last_path)
        for path in final_path_list:
            if last_path == path["path"]:
                full_path = node.path
                block_path.append({"path": full_path, "text": path["text"], "content": node.block})

        for child in node.children:
            get_full_path(child)

    get_full_path(tree)
    return block_path

# %% [markdown]
# ## Build Tree

# %%
def heuristic_function(path1, path2):
    common_prefix = 0
    for i, j in zip(path1, path2):
        if i == j:
            common_prefix += 1
        else: break
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

# %%
def build_graph(block_path):
    G = nx.DiGraph()

    for idx, path in enumerate(block_path):
        print(path['text'])
        G.add_node(" ".join(path["path"]), tag=path["path"][-1], content = path["content"], path=path["path"], embd = embd_score[idx]["embedding"])

    nodes = list(G.nodes(data="embd"))
    # print(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # nodes[i][1] = [float(x) for x in nodes[i][1]]
            # nodes[j][1] = [float(x) for x in nodes[j][1]]
            structure_rel_score = heuristic_function(nodes[i][0].split(), nodes[j][0].split())
            simi_score = similarity([nodes[i][1]], [nodes[j][1]])[0][0]
            G.add_edge(nodes[i][0], nodes[j][0], relationship="semantic", weight=simi_score * 0.5 + structure_rel_score * 0.5)


    labels = {node: G.nodes[node]['content'] for node in G.nodes}

    # Remove bottom 20% of edges by weight
    all_edges = list(G.edges(data=True))
    all_weights = [edge[2]['weight'] for edge in all_edges]
    threshold = sorted(all_weights)[int(len(all_weights) * 0.5)]  # 50% quantile

    for u, v, data in all_edges:
        if data['weight'] <= threshold:
            G.remove_edge(u, v)

    # Draw with content as labels
    nx.draw(G, with_labels=True, labels=labels, pos=nx.spring_layout(G))
    # nx.draw(G, with_labels=True)
    return G

# %%
from neo4j import GraphDatabase

# Replace with your actual info
uri = "bolt://localhost:7687"
user = "neo4j"
password = "1234567890"

def upload_graph_to_neo4j(graph, driver):
    with driver.session() as session:
        session.execute_write(_clear_existing_graph)
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


# %% [markdown]
# ## Processing Dataset

# %%

for data in ds:
    # Take data
    document = data["document"]
    question = data["question"]["text"]
    short_answer = data["annotations"]
    # long_answer_candidates = data["long_answer_candidates"]
    token_list = document["tokens"]
    html_doc = document["html"]
    print(token_list["token"])
    # Extract short answer
    short_answer_list = []
    for short_answer_element in short_answer["short_answers"]:
        if not short_answer_element["start_token"] or not short_answer_element["end_token"]:
            continue
        start_token = short_answer_element["start_token"][0]
        end_token = short_answer_element["end_token"][0]
        if start_token == -1 or end_token == -1:
            continue
        
        print(start_token, end_token)

        answer = token_list["token"][start_token: end_token]
        # for idx in range(len(token_list["token"])):
        #     # print(token_list["token"])
        #     if idx >= start_token and idx < end_token:
        #         answer.append(token_list["token"][idx])
        #     # print(answer)
        short_answer_list.append(answer)
        print(short_answer_list)

    # Clean document
    content = clean_html(html_doc)
    # content.find_all()
    dom_tree = build_dom_tree(content)
    # print_tree_contents(dom_tree)
    tree = build_block_tree(dom_tree, 500)
    
    contents_to_embd = tree_contents(tree)
    embed_contents(tree, contents_to_embd)
    # break
    print(question)
    ques_embd = fetch_embedding(question)[0]["embedding"]

    max_sim, min_sim, total_sim = similarity_compute(tree, ques_embd, 0.0, 0.0, 1.0)
    # print(max_sim, min_sim, total_sim / len(contents_to_embd))
    
    avg_sim = total_sim / len(contents_to_embd)
    pruning_tree_on_similarity(tree, ques_embd, avg_sim * 0.8)

    node_paths_and_contents_zip = prepare_data_for_llm(tree)
    block_path = reformat_llm_output(tree, node_paths_and_contents_zip, question)
    
    embd_score = fetch_embedding([path["text"] for path in block_path])

    G = build_graph(block_path)

    # driver = GraphDatabase.driver(uri, auth=(user, password))
    # upload_graph_to_neo4j(G, driver)
    # driver.close()

    # Validate final tree
    # Check if the answer locate in final tree.
    # Combine all node texts in G
    all_node_texts = " ".join([str(G.nodes[node]['content']) for node in G.nodes])

    # Check if any short answer is present in the combined text
    for answer in short_answer_list:
        answer_str = " ".join(answer)
        if answer_str in all_node_texts:
            print(f"Short answer found in graph: {answer_str}")
        else:
            print(f"Short answer NOT found in graph: {answer_str}")
    # Retrieve answer
    break


# %%
i = 3

sample1 = ds[i]["document"]["html"]
ques1 = ds[i]["question"]["text"]
token_list1 = ds[i]["document"]["tokens"]
short_answer = ds[i]["annotations"]
long_answer = ds[i]["long_answer_candidates"]

# %%
# for start_token, end_token in zip(long_answer["start_token"], long_answer["end_token"]):
#     for idx in range(len(token_list1["token"])):
#         if idx >= start_token and idx < end_token:
#             print(token_list1["token"][idx], end=" ")
        
#     print("")

# print(short_answer)


# %% [markdown]
# ## Print dataset sample


# %% [markdown]
# # Block Node

# %%
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
        return f'Tag: {self.tag}, Content: {self.content}, children: {len(self.children)}, tag: {self.tag}'

# %% [markdown]
# # Clean HTML Document

# %%
def clean_html(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Step 1: Remove <script>, <style>, and comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove attributes: (1) lengthy attributes (2) all `style` attributes
    def clean_attributes(tag):
        for attr in list(tag.attrs.keys()):
            if attr == "style" or len(str(tag[attr])) > 50:  # Remove inline styles and long attributes
                del tag[attr]

    for tag in soup.find_all(True):  # All tags
        clean_attributes(tag)

    # Step 2: Lossless Structural Compression
    # Merge single-nested tags safely
    def merge_single_nested_tags(tag):
        while (
            len(tag.contents) == 1
            and isinstance(tag.contents[0], Tag)
            and tag.contents[0].name == tag.name
        ):
            child = tag.contents[0]
            # Merge attributes safely
            for key, value in child.attrs.items():
                if key not in tag.attrs:
                    tag.attrs[key] = value
            child.unwrap()  # Unwrap instead of modifying contents directly

        for child in tag.find_all(recursive=False):  # Iterate safely
            merge_single_nested_tags(child)

    for tag in soup.find_all(True):  # All tags
        merge_single_nested_tags(tag)

    # Remove all \t and \n in the text content
    for text in soup.find_all(string=True):
        text.replace_with(text.replace("\n", "").replace("\t", ""))

    # Remove empty tags safely
    def remove_empty_tags(tag):
        for child in tag.find_all(recursive=False):
            remove_empty_tags(child)  # Recursively clean
        if not tag.contents or all(str(content).strip() in ["None",""]  for content in tag.contents):
            tag.decompose()

    remove_empty_tags(soup)

    return soup



# %% [markdown]
# ## Build DOM Tree

# %%
def build_dom_tree(html_content):
    def build_tree(element):
        if not element.name:
            return None
        
        direct_text = []
        for child in element.children:
            if isinstance(child, str):
                text = child.strip()
                if text:
                    direct_text.append(text)

        # Recursively build child nodes
        children_nodes = []
        for child in element.children:
            if hasattr(child, 'name') or isinstance(child, str):
                child_node = build_tree(child)
                if child_node:
                    children_nodes.append(child_node)
        # If no direct text, use descendant content (especially for structural tags)
        node_content = (
            " ".join(direct_text)
        )

        node = BlockNode(
            tag=element.name,
            attributes=element.attrs,
            content=node_content.strip()
        )

        for child_node in children_nodes:
            node.add_child(child_node)

        return node

    def losen_structure(node: BlockNode):
        if not node:
            return

        for child in node.children:
            losen_structure(child)

        i = 0
        while i < len(node.children):
            child = node.children[i]
            if (
                not child.content.strip()
                and len(child.children) == 1
            ):
                del node.children[i]
                node.children.insert(i, child.children[0])
            else:
                i += 1

    root = build_tree(html_content)
    losen_structure(root)
    return root


# %% [markdown]
# ## Build Block Tree

# %%
def build_block_tree(dom_tree: BlockNode, max_window: int):
    """
    Build a bottom-up balanced block tree like HTML-RAG:
    - Merge sibling leaf nodes up to max_window tokens.
    - Merge linear branch segments.
    - Retag and track paths for traceability.
    """
    
    def count_tokens(node: BlockNode) -> int:
        text = node.block or node.content or ""
        return len(text.split())
    
    def flatten_text(node: BlockNode) -> str:
        if node.block:
            return node.block.strip()
        if not node.children:
            return (node.content or "").strip()
        parts = [(node.content or "").strip()] + [flatten_text(child) for child in node.children]
        return " ".join(p for p in parts if p)

    def merge_upward(node: BlockNode, limit: int):
        """Merge leaf siblings into a single block if within limit."""
        # go to sub-leaf nodes
        # check if leaf nodes can all merge, 
        # if can merge, merge them all, and go to ancestor.
            # check if the block created have rooms for ancestor.
            # if can merge, merge them all, and go to ancestor.
            # recursively

        # else, go to ancestor
            # try to merge intermediate nodes
        if not node:
            return True
        # print("Node content: ", node.content)

        node.block = node.content if node.content else ""
        if not node.children:
            print("leaf node")
            # If no children, treat as leaf node
            node.is_leaf = True
            return node.is_leaf
        
        is_leaf = True
        for child in node.children:
            is_leaf = merge_upward(child, limit) and is_leaf
        print("is child all leaf: ", is_leaf)
        if is_leaf is False:
            # check if the node have only one child
            # true, merge the child to node and assign the child children to node children
            # false, return False
            if len(node.children) == 1:
                child = node.children[0]
                node.block = (node.content if node.content else "") + (child.block if child.block else child.content)
                print("node block: ", node.block)
                node.children = child.children
            return False
        else:
            # check if the children content and node content can be merged
            # true, merge the children content and self content to self block, return True
            # false, return False
            child_content = []
            for child in node.children:
                if not child.block:
                    if not child.content:
                        continue
                    else:
                        child_content.append(child.content)
                else:
                    child_content.append(child.block)
            token_count = len(node.content.split()) + sum(len(s.split()) for s in child_content)
            if token_count < limit:
                total_content = node.content + " " + " ".join(child_content)
                node.block = total_content
                print("node block: ", node.block)
                print("Total content: ", total_content)
                node.children = []
                node.is_leaf = True
                return True
            else:
                print("token len: ", token_count)
                # If the content exceeds the limit, keep the children as is
                if not node.block:
                    node.block = node.content or ""
                print("node block: ", node.block)
                print(len(node.children))
                node.is_leaf = False
                return False
    def retag_tree(node: BlockNode, tag_counts=None):
        if tag_counts is None:
            tag_counts = {}

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
    # Optional: enable branch merging
    # merge_branch_segments(dom_tree, max_window)

    return dom_tree


# %%
def print_tree_contents(node: BlockNode, depth=0):
    """Print tree structure with tag and block contents."""
    if not node:
        return
    print("    " * depth + f"Tag: {node.tag}, Content: {node.content}, Attr:{node.attributes}")
    for child in node.children:
        # print(child.content)
        print_tree_contents(child, depth + 1)

# %%
def print_tree_blocks(node: BlockNode, indent: int = 0, with_path = False):
    """Print the block tree."""
    if with_path is True:
        print("    " * indent + f"{node.block}", end="")
        if node.path != []:
            print(node.path)
        else:
            print("dont have path")
    else:
        # print("    " * indent + f"{node.block}")
        print(node.block)
        print(node.content)
    # print(len(node.children))
    for child in node.children:
        print_tree_blocks(child, indent + 1, with_path)

# %% [markdown]
# ## Apply the building

# %%
content = clean_html(sample3)
# content.find_all()
dom_tree = build_dom_tree(content)
# print_tree_contents(dom_tree)
tree = build_block_tree(dom_tree, 500)
print_tree_blocks(tree, with_path=True)

# %% [markdown]
# # Pruning the tree

# %%
EMB_MODEL_ID = "text-embedding-nomic-embed-text-v1.5-embedding"

def fetch_embedding(text):
    """Fetch embedding for a given text from LMStudio."""
    try:
        url = "http://localhost:9999/v1/embeddings"
        data = {"model": EMB_MODEL_ID, "input": text}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            # print(response.json())
            return response.json()["data"]
        else:
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error querying LMStudio: {e}")
        return None

# %%
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

# %%
def print_tree_embeddings(node: BlockNode, depth=0):
    """Print tree structure with embeddings."""
    if not node:
        return
    # if node.content.strip():
    print(
        "  " * depth
        + f"Tag: {node.tag}, Text: {node.block}, Embeddings: {node.embedding}"
    )
    for child in node.children:
        print_tree_embeddings(child, depth + 1)

# %%
def embed_contents(root: BlockNode, contents):
    embd = fetch_embedding(contents)

    # print(len(embd))
    def tranverse_tree(root: BlockNode, idx):
        if not root:
            return
        if root.block.strip():
            root.embedding = embd[idx]["embedding"]
            # print(root.content, idx, root.embedding)
            idx += 1
        for child in root.children:
            idx = tranverse_tree(child, idx)
            # print(idx)
        return idx

    tranverse_tree(root, 0)

# %% [markdown]
# ## Embedding the block tree contents

# %% [markdown]
# ### Create the tree contents

# %%
contents_to_embd = tree_contents(tree)


# %% [markdown]
# ## Embeddings the contents

# %%
embed_contents(tree, contents_to_embd)

# %%
# ques1 = "Who proposed the idea that particles can exhibit wave-like behavior?"

# %%
ques3 = "Phòng tốt nhất là phòng nào?"

# %% [markdown]
# ## Embedding the question

# %%
ques_embd = fetch_embedding(ques3)[0]["embedding"]

# %% [markdown]
# ## Calculate the Cos-sim value of contents and question

# %%
def similarity_compute(node: BlockNode, ques_embd, total_sim, max_sim, min_sim):
    if not node:
        return
    if node.embedding is not None:
        cosine_sim = similarity([node.embedding], [ques_embd])[0][0]
        print(cosine_sim, node.block) if cosine_sim > 0.5 else None
        max_sim = max(max_sim, cosine_sim)
        min_sim = min(min_sim, cosine_sim)
        total_sim += cosine_sim

    for child in node.children:
        max_sim, min_sim, total_sim = similarity_compute(
            child, ques_embd, total_sim, max_sim, min_sim
        )
    return max_sim, min_sim, total_sim

# %%
max_sim, min_sim, total_sim = similarity_compute(tree, ques_embd, 0.0, 0.0, 1.0)
print(max_sim, min_sim, total_sim / len(contents_to_embd))

# %% [markdown]
# ## Pruning block tree base on computed cos-sim

# %%
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

# %%
avg_sim = total_sim / len(contents_to_embd)
pruning_tree_on_similarity(tree, ques_embd, avg_sim * 0.9)

# %% [markdown]
# ## Pruning with LLM

# %% [markdown]
# ### Adding the path to the block tree 

# %%
def get_node_paths_and_contents(node: BlockNode, path="") -> list:
    """Return a list containing each node path and node block content."""
    if not node:
        return []
    # print(node)
    current_path = f"<{node.path[-1]}>"
        
    result = [{"path": current_path, "content": node.block}]
    
    if not node.block.strip():
        result.pop()
    for child in node.children:
        result.extend(get_node_paths_and_contents(child, current_path))
    return result

# # Get the list of node paths and contents
node_paths_and_contents = get_node_paths_and_contents(tree)
# print(node_paths_and_contents)
node_paths_and_contents_zip = [[node_paths_and_contents[0]['path']+node_paths_and_contents[0]['content']]]
for path in node_paths_and_contents[1:]:
    if len(path["path"]) + len(path["content"]) < 1000 - len(''.join(node_paths_and_contents_zip[-1])):
        node_paths_and_contents_zip[-1].append(path["path"]+path["content"])
    else:
        node_paths_and_contents_zip.append([path["path"]+path["content"]])
node_paths_and_contents_zip

# %%
def count_nodes(node: BlockNode) -> int:
    if not node:
        return 0
    count = 1  # Count the current node
    for child in node.children:
        count += count_nodes(child)
    return count


# # Count the number of nodes in the latest tree
# num_nodes = count_nodes(tree)
# print(f"Number of nodes in the tree: {num_nodes}")

# %% [markdown]
# ## Call Groq API to get node relationship

# %%
client = Groq(
    # This is the default and can be omitted
    api_key=os.getenv("GROQ_API"),
)


def inference(html_content, question):
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
                **HTML**: {HTML tag and content}.
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
                    "content": f'**HTML**: "{html_content}"\n**Question**: "{question}"\nPlease output in JSON format. {{"relevant_blocks": []}}',
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
                    "strict": ["relevant_blocks"]
                },
            },
            model="llama3-70b-8192",
            # model="llama",
            temperature=0.5,
            stream=False,
            max_tokens=2000,
        )

    except Exception as e:
        print(f"Error: {e}")
        return None
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

# %%
raw_answer = []
for path in node_paths_and_contents_zip:
    min_items = min(len(path), 5)
    path_response = inference(path, ques3)
    
    if path_response:
        raw_answer.append(path_response)

    time.sleep(5)


# %% [markdown]
# ### Clean the LLM output

# %%
final_path_list = []
for path in raw_answer:
    final_path_list.extend(json.loads(path)["relevant_blocks"])
for path in final_path_list:
    print(path)

# %%
for path in final_path_list:
    element = path["path"].split(">")
    miss_parse = 0
    if len(element) > 1:
        for elem in element[:-1]:
            if not elem.startswith("<"):
                print(path["text"], elem)
                path["text"] = path["text"] + elem if elem not in path["text"] else ""
            else:
                miss_parse += 1
        # print(element, miss_parse)
        if "text" not in path:
            path["text"] = ""
        path["text"] += (" " + element[-1]) if element[-1] not in path["text"] else ""
        path["path"] = element[miss_parse - 1] + ">"
    print(path)

# %%
block_path = []
def get_full_path(node: BlockNode):
    if not node:
        return
    last_path = f"<{node.path[-1]}>"
    # print(last_path)
    for path in final_path_list:
        if last_path == path["path"]:
            full_path = node.path
            block_path.append({"path": full_path, "text": path["text"], "content": node.block})

    for child in node.children:
        get_full_path(child)

# %%
get_full_path(tree)

# %% [markdown]
# ## heuristic evaluation for the block tree content relationship

# %%
def heuristic_function(path1, path2):
    common_prefix = 0
    for i, j in zip(path1, path2):
        if i == j:
            common_prefix += 1
        else: break
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

# %%
embd_score = fetch_embedding([path["text"] for path in block_path])

# %% [markdown]
# # Plot the tree

# %%
G = nx.DiGraph()

for idx, path in enumerate(block_path):
    print(path['text'])
    G.add_node(" ".join(path["path"]), tag=path["path"][-1], content = path["content"], path=path["path"], embd = embd_score[idx]["embedding"])

nodes = list(G.nodes(data="embd"))
# print(nodes)
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        # nodes[i][1] = [float(x) for x in nodes[i][1]]
        # nodes[j][1] = [float(x) for x in nodes[j][1]]
        structure_rel_score = heuristic_function(nodes[i][0].split(), nodes[j][0].split())
        simi_score = similarity([nodes[i][1]], [nodes[j][1]])[0][0]
        G.add_edge(nodes[i][0], nodes[j][0], relationship="semantic", weight=simi_score * 0.5 + structure_rel_score * 0.5)


labels = {node: G.nodes[node]['content'] for node in G.nodes}

# Remove bottom 20% of edges by weight
all_edges = list(G.edges(data=True))
all_weights = [edge[2]['weight'] for edge in all_edges]
threshold = sorted(all_weights)[int(len(all_weights) * 0.5)]  # 50% quantile

for u, v, data in all_edges:
    if data['weight'] <= threshold:
        G.remove_edge(u, v)

# Draw with content as labels
nx.draw(G, with_labels=True, labels=labels, pos=nx.spring_layout(G))
# nx.draw(G, with_labels=True)

# %%
len(G.edges)

# %%
from neo4j import GraphDatabase

# Replace with your actual info
uri = "bolt://localhost:7687"
user = "neo4j"
password = "1234567890"

# %%
driver = GraphDatabase.driver(uri, auth=(user, password))

# %%
def upload_graph_to_neo4j(graph):
    with driver.session() as session:
        session.execute_write(_clear_existing_graph)
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


# %%

# Run the upload
upload_graph_to_neo4j(G)
driver.close()

# %%
nodes = list(G.nodes(data="content"))
data_for_GPT = []
for u, v, data in G.edges(data=True):
    u_content = ""
    v_content = ""
    for node in nodes:
        if node[0] == u:
            u_content = node[1]
        if node[0] == v:
            v_content = node[1]

    data_for_GPT.append((u, u_content, v, v_content))


# %%
def edge_inference(list_of_tuple):
    message_content = []
    batch_size = len(list_of_tuple)
    for elem in list_of_tuple:
        message_content.append(
            f'{{\n"Path 1": "{elem[0]}",\n"Content 1": "{elem[1]}",\n"Path 2": "{elem[2]}",\n"Content 2": "{elem[3]}"\n}}\n'
        )
    message_content = "["+",".join(message_content)+"]"
    print(message_content)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
            You are a highly intelligent system trained to analyze relationships between HTML tree paths and determine the likelihood that one path supports or complements the information of the other.

            Each path represents a location in a document's structure, and the relationship is influenced by their content, position, and context.

            Your task is to:
            1. Evaluate the semantic and structural relationship between multiple pairs of paths.
            2. For each pair, return a single float score between 0.0 and 10.0, where:
            - 0.0 means the paths are completely unrelated or conflicting.
            - 10.0 means the paths strongly support each other.
            3. The number of scores MUST **exactly match** the number of input pairs.
            4. Ensure your output strictly follows this format:
                ```json
                {
                    "evaluation": [
                        {"score": <float>},
                        {"score": <float>},
                        {"score": <float>}
                    ]
                }
            **Input Format**:
            A list of string like JSON object, where each object contains:
            - **Path 1**: "Path of first block content"
            - **Content 1**: "Content of first block"
            - **Path 2**: "Path of second block content"
            - **Content 2**: "Content of second block"

            **Output Format**:
            A list of dictionaries, where each dictionary contains:
            - **Score**: [float value from 0.0 to 10.0],

            **Example**:

            **Input**:
            [
                {
                    "Path 1": "document1 html1 body1 div1 div4 div7 div8",
                    "Content 1": "Annual events History Personalities Awards and honors OthersNational Basketball Associationshow National Basketball Association",
                    "Path 2": "document1 html1 body1 div1 div4 div7 div8 div48 table8 tbody8 tr104 td343 div51 ul13 li89",
                    "Content 2": "Jordan"
                },
                {
                    "Path 1": "document2 html2 body2 div3 div6 div9",
                    "Content 1": "AI and Machine Learning Research Trends",
                    "Path 2": "document2 html2 body2 div3 div6 div9 div12 p4",
                    "Content 2": "Deep Learning advances in 2023"
                },
                {
                    "Path 1": "document2 html2 body2 div2 div5 ul1 li3",
                    "Content 1": "Top 10 programming languages in 2025",
                    "Path 2": "document2 html2 body2 div2 div5 ul1 li3 span1",
                    "Content 2": "Python"
                },
                {
                    "Path 1": "document3 html3 body3 section1 article2 p5",
                    "Content 1": "Upcoming advancements in renewable energy technologies",
                    "Path 2": "document3 html3 body3 section1 article2 p7",
                    "Content 2": "Battery storage and smart grids"
                }
            ]

            **Output**:
            {
                "evaluation": [
                    {"Score": 3.776},
                    {"Score": 9.123},
                    {"Score": 9.99},
                    {"Score": 8.41}
                ]
            }

            Evaluate the relationships and provide your output in JSON with the following format.
            """,
            },
            {
                "role": "user",
                "content": f"""Here is input for {batch_size} pairs of HTML tree paths:
                **Input**: 
                {message_content}
                Provide exact {batch_size} scores coresponding to the input.
                """,
            },
        ],
        response_format={
            "type": "json_object",  # Use 'json_object' as the allowed type
            "json_object": {
                "evaluation": [
                    {"score": "float"},
                ]
            },
        },
        model="llama-3.1-8b-instant",
        # temperature=0.8,
        stream=False,
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

# %%
# store_data = data_for_GPT.copy()
# len(store_data)

# %%
# data_for_GPT = store_data.copy()

# GPT_scoring = []
# while len(data_for_GPT) > 0:
#     batch_size = 20
#     GPT_scoring.append(edge_inference(data_for_GPT[:batch_size]))
#     data_for_GPT = data_for_GPT[batch_size:]

#     time.sleep(6)

# %%
# edge_score = []

# for raw_score in GPT_scoring:

#     raw_score = raw_score.lower()

#     process_score = json.loads(raw_score)["evaluation"]
#     for score in process_score:
#         edge_score.append(float(score["score"]))
    

# %% [markdown]
# # Retrieve Answer from tree block

# %%
def retrieve_answer_from_graph(G: nx.DiGraph, question: str, ques_embd):
    score_rank = []
    for node, data in G.nodes(data=True):
        score_rank.append(similarity([data["embd"]], [ques_embd])[0][0])
        
    top5_idx = np.argsort(score_rank)[-5:][::-1]
    top5_nodes = [list(G.nodes)[idx] for idx in top5_idx]
    print(len(top5_nodes))
    # Step 3: Traverse the graph starting from the top 5 nodes
    visited = set()
    answers = []

    def traverse(node, accumulated_weight=0):
        # Avoid revisiting nodes
        if node in visited:
            return
        visited.add(node)
        print(G.nodes[node]['content'])
        # Update node rank with accumulated weight
        score_rank[list(G.nodes).index(node)] += accumulated_weight
        print(accumulated_weight)
        # Collect answer if node is relevant
        answers.append((G.nodes[node]['content'], score_rank[list(G.nodes).index(node)]))

        # Sort outgoing edges by weight and traverse the most relevant ones
        neighbors = list(G.neighbors(node))
        neighbors.sort(key=lambda n: G[node][n]['weight'], reverse=True)
        print("neighbors: ", neighbors)
        for neighbor in neighbors[:10]:
            traverse(neighbor, accumulated_weight=G[node][neighbor]['weight'] * 0.9 + accumulated_weight * 0.1)

    # Start traversal from the top 5 nodes
    for start_node in top5_nodes:
        traverse(start_node)
    # Start traversal from the top 5 nodes
    for start_node in top5_nodes:
        print(start_node)
        traverse(start_node)


    # Step 4: Aggregate answers and return the best one
    if answers:
        # Sort answers by relevance score
        answers.sort(key=lambda x: x[1], reverse=True)

        return answers  # Return the text of the most relevant node
    else:
        return "No relevant answer found."
    
answer = retrieve_answer_from_graph(G, ques3, ques_embd)

# %%
retrieve_ans = ""
for ans in answer:
    retrieve_ans += ans[0] + " / "

# %%
retrieve_ans

# %% [markdown]
# ## Load the retrieved answer to the LLM Model to result the final answer

# %%
def retrieve_(additional_content, question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
            You are a helpful assistant responsible for summarizing information and using knowledge to answering question.
            Given a string of many informations, and a question, all related to that question.
            Please answer the question using your knowledge with the support of information provided.
            If the provided informations does not support the answering process, please resolve the informations and answer the question base on your knowledge.
            
            ### INPUT ###
            "Additional Information": Information.
            "Question": Question.
            
            ### OUTPUT ###
            "Answer": Answer of the given question use your knowledge with additional information.
            """,
            },
            {
                "role": "user",
                "content": f"""
                Additional Information: {additional_content}
                Question: {question}
                Please provide the answer in JSON format {{"answer": Your Answer}}
                """,
            },
            
        ],
        response_format={
            "type": "json_object",  # Use 'json_object' as the allowed type
            "json_object": {
                "answer": [
                    {"answer": "string"},
                ]
            },
        },
        model="deepseek-r1-distill-llama-70b",
        temperature=0.1,
        stream=False,
    )
    print(question)
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

# %%
final_answer = json.loads(retrieve_(retrieve_ans, ques3))["answer"]

# %%
final_answer

# %%
short_answer

# %%
def nearly_match(final_answer, short_answers):
    client = Groq(api_key=os.getenv("GROQ_API"))

    def compare_answers(answer1, answer2):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                    You are an expert system designed to evaluate the similarity between two answers. Your task is to determine if the two provided answers are nearly the same in meaning, even if they are not exactly identical in wording.
                    Please provide a score between 0 and 1, where 1 means the answers are nearly identical in meaning, and 0 means they are completely different.
                    """,
                    },
                    {
                        "role": "user",
                        "content": f'Answer 1: "{answer1}"\nAnswer 2: "{answer2}"\nPlease provide the similarity score in JSON format. Output: {{"score": }}',
                    },
                ],
                response_format={
                    "type": "json_object",
                    "json_object": {
                        "score": "float",
                    },
                },
                model="llama-3.1-8b-instant",
                temperature=0.2,
                stream=False,
                max_tokens=50,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

    for answer in short_answers:
        if answer["text"] == []:
            continue
        answer_text = " ".join(answer["text"])
        similarity_score = json.loads(compare_answers(final_answer, answer_text))
        print(similarity_score)
        if (
            similarity_score and float(similarity_score["score"]) > 0.8
        ):  # Threshold for near match
            return True
    return False


# Example usage
# short_answers = short_answer["short_answers"]
# is_nearly_match = nearly_match(final_answer, short_answers)
# print(is_nearly_match)

# %%


# %%
def exact_match(final_answer, short_answers):
    for answer in short_answers:
        if answer['text'] == []:
            continue
        print(answer)
        if final_answer.strip().lower() == answer['text'][0].strip().lower():
            return True
    return False

# Example usage
# short_answers = short_answer['short_answers']
# is_exact_match = exact_match(final_answer, short_answers)
# print(is_exact_match)

# %%
# Loop through the dataset and process each item
for i in range(1):
    # Extract the necessary data
    sample_html = ds[i]["document"]["html"]
    question = ds[i]["question"]["text"]
    token_list = ds[i]["document"]["tokens"]
    short_answer = ds[i]["annotations"]
    long_answer = ds[i]["long_answer_candidates"]

    # Clean the HTML content
    cleaned_content = clean_html(sample_html)

    # Build the DOM tree
    dom_tree = build_dom_tree(cleaned_content)

    # Build the block tree
    block_tree = build_block_tree(dom_tree, 10)
    # Get the contents to embed
    contents_to_embed = tree_contents(block_tree)

    # Fetch embeddings for the contents
    embed_contents(block_tree, contents_to_embed)
    print_tree_blocks(block_tree)
    
    # Fetch embedding for the question
    question_embedding = fetch_embedding(question)[0]["embedding"]

    # Compute similarity
    max_sim, min_sim, total_sim = similarity_compute(block_tree, question_embedding, 0.0, 0.0, 1.0)

    # Prune the tree based on similarity
    pruning_tree_on_similarity(block_tree, question_embedding, total_sim / len(contents_to_embed) * 0.1)
    print(block_tree)
    # Get node paths and contents
    node_paths_and_contents = get_node_paths_and_contents(block_tree)

    # Zip the node paths and contents
    node_paths_and_contents_zip = [[node_paths_and_contents[0]['path'] + node_paths_and_contents[0]['content']]]
    for path in node_paths_and_contents[1:]:
        if len(path["path"]) + len(path["content"]) < 1000 - len(''.join(node_paths_and_contents_zip[-1])):
            node_paths_and_contents_zip[-1].append(path["path"] + path["content"])
        else:
            node_paths_and_contents_zip.append([path["path"] + path["content"]])

    # Call Groq API to get node relationship
    raw_answer = []
    for path in node_paths_and_contents_zip:
        min_items = min(len(path), 5)
        path_response = inference(path, question, min_items)
        if path_response:
            raw_answer.append(path_response)
        time.sleep(1)

    # Clean the LLM output
    final_path_list = []
    for path in raw_answer:
        final_path_list.extend(json.loads(path)["relevant_blocks"])

    for path in final_path_list:
        element = path["path"].split(">")
        miss_parse = 0
        if len(element) > 1:
            for elem in element[:-1]:
                if not elem.startswith("<"):
                    print(path["text"], elem)
                    path["text"] = path["text"] + elem if elem not in path["text"] else ""
                else:
                    miss_parse += 1
            # print(element, miss_parse)
            if "text" not in path:
                path["text"] = ""
            path["text"] += (" " + element[-1]) if element[-1] not in path["text"] else ""
            path["path"] = element[miss_parse - 1] + ">"
        # print(path)
        
    # Get full path
    block_path = []
    get_full_path(block_tree)

    # Fetch embeddings for block path
    embd_score = fetch_embedding([path["text"] for path in block_path])

    # Create the graph
    G = nx.DiGraph()
    for idx, path in enumerate(block_path):
        G.add_node(" ".join(path["path"]), tag=path["path"][-1], content=path["text"], path=path["path"], embd=embd_score[idx]["embedding"])

    nodes = list(G.nodes(data="embd"))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            structure_rel_score = heuristic_function(nodes[i][0].split(), nodes[j][0].split())
            simi_score = similarity([nodes[i][1]], [nodes[j][1]])[0][0]
            G.add_edge(nodes[i][0], nodes[j][0], relationship="semantic", weight=simi_score * 0.3 + structure_rel_score * 0.7)

    # Retrieve answer from tree block
    answer = retrieve_answer_from_graph(G, question, question_embedding)

    # Print the final answer
    print(answer)

    retrieve_ans = ""
    for ans in answer:
        retrieve_ans += ans[0] + " / "

    final_answer = json.loads(retrieve_(retrieve_ans, question))["answer"]
    short_answers = short_answer['short_answers']
    is_exact_match = exact_match(final_answer, short_answers)
    print(is_exact_match)

# %%


# %% [markdown]
# ###For Testing:
# 
#   1. Wave–particle duality
# 
#   Q: What is wave–particle duality?
#   A: Wave–particle duality is a fundamental concept in quantum mechanics, which states that particles, such as electrons, can exhibit both wave-like and particle-like behavior depending on how they are observed.
# 
#   Q: Which experiment demonstrates the wave nature of particles?
#   A: The Young's Double Slit Experiment and the Afshar experiment, as well as decoherence of matter waves by thermal emission of radiation, real-time single-molecule imaging of quantum interference, and the Talbot Lau interferometer demonstrate the wave nature of particles..
# 
#   Q: Who proposed the idea that particles can exhibit wave-like behavior?
#   A: Louis de Broglie proposed the idea that particles can exhibit wave-like behavior, and Max Planck, Albert Einstein, Arthur Compton, and Niels Bohr also contributed to the understanding of wave-particle duality..
# 
#   2. List of current United States senators
# 
#   Q: How many U.S. senators are there in total?
#   A: There are 100 senators in the US Senate, with 47 Democrats and 53 Republicans. The remaining 3 are Independents.
# 
#   Q: What is the term length for a U.S. senator?
#   A: The term length for a U.S. senator is 6 years.
# 
#   3. Deposition (phase transition)
# 
#   Q: What is deposition in terms of phase transition?
#   A: Deposition is the phase transition in which a gas transforms directly into a solid without passing through the liquid phase.
# 
#   Q: What is an example of deposition occurring in nature?
#   A: Frost formation on surfaces is an example of deposition, where water vapor turns directly into ice.
# 
#   Q: What is the reverse process of deposition called?
#   A: The reverse process of deposition is called sublimation.
# 
#   4. Longest word in English
# 
#   Q: What is the longest word in a major English dictionary?
#   A: "Pneumonoultramicroscopicsilicovolcanoconiosis" is the longest word in a major English dictionary.
# 
#   Q: What does "antidisestablishmentarianism" mean?
#   A: "Antidisestablishmentarianism" refers to opposition to the disestablishment of the Church of England.
# 
#   Q: Which word is known for being coined to have many letters but is not technical?
#   A: "Supercalifragilisticexpialidocious" is a coined word known for its length and was popularized by the film "Mary Poppins."
# 
#   5. NBA All-Star Game Kobe Bryant Most Valuable Player
# 
#   Q: What is the NBA All-Star Game Kobe Bryant Most Valuable Player award?
#   A: It is an annual award given to the most outstanding player of the NBA All-Star Game, renamed in honor of Kobe Bryant in 2020.
# 
#   Q: Who holds the record for the most NBA All-Star MVP awards?
#   A: Kobe Bryant and LeBronJames are tied, each having won the award 4 times, following up is Michael Jordan with 3.
# 
#   Q: When was the NBA All-Star MVP award first introduced?
#   A: The NBA All-Star MVP award was first introduced in 1953.
# 
#   6. Water distribution on Earth
# 
#   Q: What percentage of Earth's water is freshwater?
#   A: Approximately 2.5% of Earth's water is freshwater.
# 
#   Q: Where is the majority of Earth's freshwater stored?
#   A: The majority of Earth's freshwater is stored in glaciers and ice caps.
# 
#   Q: What fraction of Earth's freshwater is accessible for human use?
#   A: Only about 1% of Earth's freshwater is easily accessible for human use, found in lakes, rivers, and shallow groundwater.
# 
#   7. Who Wants to Be a Millionaire? New Zealand
# 
#   Q: Who hosted the New Zealand version of "Who Wants to Be a Millionaire?"
#   A: Mike Hosking hosted the New Zealand version of the show.
# 
#   Q: In what year did "Who Wants to Be a Millionaire? New Zealand" first air?
#   A: It first aired in 2008.
# 
#   Q: Where was the New Zealand version of the show filmed?
#   A: It was filmed in Melbourne, Australia, on the set of the Australian version.
# 
#   8. Brock
# 
#   Q: In the Pokémon series, who is Brock?
#   A: Brock is a character who is the Pewter City Gym Leader and a companion of Ash Ketchum.
# 
#   Q: What is Brock University?
#   A: Brock University is a public university located in St. Catharines, Ontario, Canada.
# 
#   Q: Who is Eddie Brock in Marvel Comics?
#   A: Eddie Brock is a character who becomes the host for the alien symbiote known as Venom.
# 
#   9. pH
# 
#   Q: What does pH measure?
#   A: pH measures the acidity or alkalinity of a solution.
# 
#   Q: What is the pH value of pure water at 25°C?
#   A: The pH value of pure water at 25°C is approximately 7, which is neutral.
# 
#   Q: Who introduced the concept of pH?
#   A: The concept of pH was introduced by Danish chemist Søren Peder Lauritz Sørensen in 1909.
# 
#   10. Edible bird's nest
# 
#   Q: What is an edible bird's nest made from?
#   A: An edible bird's nest is made from the solidified saliva of swiftlets.
# 
#   Q: In which cuisine are edible bird's nests considered a delicacy?
#   A: They are considered a delicacy in Chinese cuisine.
# 
#   Q: What is the primary dish prepared using edible bird's nests?
#   A: The primary dish is bird's nest soup.

# %%



