AgenticiRouterKGSysPrompt = """You are a knowledge graph construction expert. Your task is to help build a hierarchical classification system for an intelligent model router called **AgenticiRouter**.

AgenticiRouter selects the most appropriate Large Language Model (LLM) based on a user's query and preferences (e.g., performance, cost-efficiency, latency). To support this, we are building a multi-level taxonomy of query types.

Each query is classified into a four-level node path:

1. Domain — broad task area 
2. Subcategory  — specific task type 
3. Difficulty Level (optional) — complexity level within that task type 
4. User Preference (optional) - desired output behavior

"""

MathDomain = {
    "name": "Mathematics",
    "definition": "Covers quantitative problem-solving tasks involving numbers, equations, logic, or formal systems, including arithmetic, algebra, calculus, and more.",
    "example": "What is the derivative of sin(x)?"
}

CreativeWritingDomain = {
    "name": "Creative Writing",
    "definition": "Involves imaginative or artistic language generation tasks such as writing poems, stories, scripts, or creative descriptions.",
    "example": "Write a short story about a robot who learns to paint."
}

CommonsenseKnowledgeDomain = {
    "name": "Commonsense Knowledge",
    "definition": "Tasks requiring everyday world knowledge or intuitive reasoning about physical, social, or causal relationships.",
    "example": "Can a person fit inside a microwave?"
}


ProgrammingDomain = { 
    "name": "Programming",
    "definition": "Involves tasks related to code generation, debugging, explanation, or software development using programming languages.",
    "example": "Fix the bug in this Python function that calculates factorial."
}

LongFormQuestionAnsweringDomain = {
    "name": "Long-context Understanding",
    "definition": "Tasks that involve processing, understanding, and reasoning over long user inputs or extended context passages, such as multi documents, or transcripts.",
    "example": "Given the full meeting transcript, summarize the key action items and decisions."
}

ReadingComprehensionDomain = {
    "name": "Reading Comprehension",
    "definition": "Tasks that involve understanding and interpreting a given passage or document to answer specific questions.",
    "example": "Based on the article, what was the author's attitude toward climate change policy?"
}

LegalDomain = {
"name": "Legal Reasoning",
"definition": "Covers tasks involving legal reasoning, statutory interpretation, case law analysis, and application of legal principles. Includes question answering and reasoning tasks similar to those in the LegalBench dataset.",
"example": "According to contract law principles, can a unilateral mistake render a contract voidable?"
}

MedicalQADomain = {
"name": "Medical Knowledge",
"definition": "Involves medical knowledge recall and clinical reasoning tasks such as diagnosis, treatment guidelines, and anatomy or physiology questions, similar to those in the MedMCQA dataset.",
"example": "A patient presents with fatigue, pallor, and spoon‑shaped nails. What is the most likely deficiency?"
}

DomainNodePrompt = AgenticiRouterKGSysPrompt + """
**Step 1: Generate Domain Nodes**

Generate a list of distinct, non-overlapping **Domain** categories that represent broad types of user tasks.

We have already defined the following six initial Domains:

- Mathematics  
- Creative Writing  
- Commonsense Knowledge 
- Programming 
- Long-context Understanding 
- Reading Comprehension

Please propose **4 new high-level Domains** that:

**Rules:**

{DomainNodeRule}

**Output Format:**

For each proposed Domain:
- Name the Domain
- Provide a one-sentence definition
- Include a real-world example query

**Example:**
<node begin>
Domain: Mathematics
Definition: Covers quantitative problem-solving tasks involving numbers, equations, logic, or formal systems, including arithmetic, algebra, calculus, and more.
Example: What is the derivative of sin(x)?

Domain: Creative Writing
Definition: Involves imaginative or artistic language generation tasks such as writing poems, stories, scripts, or creative descriptions.
Example: Write a short story about a robot who learns to paint.
</node end>

**Output:**
"""

DomainNodeRule =  """
- Domains must be general yet semantically distinct.
- Avoid overlapping or ambiguous categories.
- Think of areas commonly found in LLM benchmarks (e.g., MMLU, AGIEval, CMMLU) and real-world applications.
- These will serve as the top-level nodes of the taxonomy.
- Only up to 4 new domains can be proposed.
""" 

SubcategoryNodePrompt = AgenticiRouterKGSysPrompt + """
### Step 2: Generate Subcategories for a Given Domain

The current Domain is: {domain_name}
Domain Definition: {domain_definition}
Domain Example: {domain_example}

Please generate a list of fine-grained Subcategories Nodes that represent specific types of tasks or problem types within this domain.

Please propose **up to {max_subcategory_nodes} new Subcategory Nodes** for each domain node that:

**Rules:**

{SubcategoryNodeRule}

**Output Format:**
For each proposed Domain:
- Name the Subcategory
- Provide a one-sentence definition
- Include a real-world example query


Example: 

Assume the Domain is:

Domain_name: Mathematics
Domain Definition: Covers quantitative problem-solving tasks involving numbers, equations, logic, or formal systems, including arithmetic, algebra, calculus, and more.
Domain Example: What is the derivative of sin(x)?

Then the Subcategory Nodes are:

<node begin>
Subcategory: Arithmetic
Definition: Covers basic arithmetic operations and problem-solving involving numbers, including addition, subtraction, multiplication, and division.
Example: What is the sum of 123 and 456?

Subcategory: Algebra
Definition: Involves solving equations and expressions using variables and algebraic manipulation.
Example: Solve for x in the equation 2x + 3 = 11.

...
</node end>

**Output:**:
"""

SubcategoryNodeRule = """
- Avoid overlap between subcategories.
- Each subcategory should represent a common type of user query that can be grouped under this domain.
- Subcategories should be general enough to cover many user queries but specific enough to guide model selection.
- Only up to {max_subcategory_nodes} new Subcategory Nodes can be proposed for each domain node.
"""

DifficultyLevelNodePrompt = AgenticiRouterKGSysPrompt + """
### Step 3: Define Difficulty Levels for a Given Subcategory

You are now working on level 3: **Difficulty Level**. This level captures how task complexity varies **within a specific Subcategory**.

Please utilize a fixed global scale (e.g., Easy / Medium / Hard for three-level difficulty) as the short name of each level.  
However, the Definition of each level should be customized based on the nature of the specific Subcategory.

**Input Subcategory Information:**
- Domain: {domain_name}  
- Subcategory: {subcategory_name}  
- Subcategory Definition: {subcategory_definition}  
- Subcategory Example Query: {subcategory_example}

Please propose **up to {max_difficulty_level_nodes} Difficulty Level Nodes** that:

**Rules:**

{DifficultyLevelNodeRule}

**Output Format:**

For each proposed level:
- Name the Level
- Provide a one-sentence definition
- Include a real-world example query

**Example**(Subcategory: Code Debugging): 
Assume the Subcategory is  Code Debugging.
Definition: Identifying and fixing errors in code snippets written in common programming languages.  

Then the Subcategory Nodes are:
<node begin>
Level: Basic
Definition: Single-line or syntax-only bugs in short, self-contained functions.
Example: Fix the indentation error in this 5-line Python function.

Level: Intermediate
Definition: Logic bugs in medium-length code that require control flow analysis.
Example: Fix the loop condition that causes an infinite loop in this JavaScript function.

Level: Advanced
Definition: Bugs across multiple functions, handling of edge cases, or requiring domain-specific knowledge.
Example: Fix the bug in this Flask app that breaks when uploading empty files.

Level: Expert
Definition: Deep reasoning over large codebases, concurrency, or memory management.
Example: Fix the deadlock issue in this multi-threaded C++ program handling file writes.
</node end>

**Output:**
"""

DifficultyLevelNodeRule = """- Levels must be ordered from easiest to hardest.
- Levels should reflect increasing reasoning complexity, token usage, or LLM capability required.
- Levels should be mutually exclusive — no query should belong to more than one level.
- Levels should be collectively exhaustive — all queries in this subcategory must be covered.
- Avoid generic differences like “longer text” unless it reflects actual difficulty in reasoning or generation.
- Only up to {max_difficulty_level_nodes} Difficulty Level Nodes can be proposed for each Subcategory Node.
"""

UserPreferenceNodePrompt = AgenticiRouterKGSysPrompt + """
### Step 4: Define User Preference Nodes for a Given Subcategory

You are now working on level 4: **User Preference**. This level captures **output-related constraints** or **stylistic preferences** expressed or implied in a user query. These preferences may influence output **length** and are essential for model routing.

**Input Subcategory Information:**
- Domain: {domain_name}  
- Subcategory: {subcategory_name}  
- Subcategory Definition: {subcategory_definition}  
- Subcategory Example Query: {subcategory_example}  

Please propose **up to {max_user_preference_nodes} User Preference Nodes** that:

**Rules:**

{UserPreferenceNodeRule}

**Output Format:**

For each preference, provide:
- Preference: A short label (e.g., Concise Answer, Step-by-step Explanation)  
- Definition: One sentence explaining what this preference means  
- Example: A real-world query that reflects this preference

**Example**(Subcategory: Scientific Explanation)): 
Assume the Subcategory is Scientific Explanation.
Definition: Tasks that require explaining scientific phenomena, principles, or terms in natural language.  

Then the Subcategory Nodes are:
<node begin>
Preference: Concise Explanation  
Definition: The user wants a brief, high-level answer without detailed elaboration.  
Example: Give me a short answer: Why does ice float?

Preference: Step-by-step Reasoning  
Definition: The user wants a logical, sequential breakdown of the explanation.  
Example: Explain step by step how rain forms from water vapor.

Preference: Simplified Language  
Definition: The user expects the response to avoid jargon and use plain language.  
Example: Explain quantum entanglement in the simplest way possible.

Preference: No Explicit Preference  
Definition: The user does not specify output constraints; default explanation is acceptable.  
Example: What causes ocean tides?"
</node end>

**Output:**
"""

UserPreferenceNodeRule = """- Represent common, realistic user preferences for how the output should be generated.
- Are **mutually exclusive** — each preference describes a distinct user intent.
- Are **semantically distinct** — combine similar preferences when appropriate to avoid redundancy.
- Are **collectively useful** — together they should cover the major output styles or expectations for this Subcategory.
- Should significantly influence LLM's generation length.
- Must include a special node for **“No Explicit Preference”** (i.e., when a user query does not express any specific output-related constraints).
- Only up to {max_user_preference_nodes} User Preference Nodes can be proposed for each Subcategory Node.
"""

NodeRevisePrompt = AgenticiRouterKGSysPrompt + """
**Sepcial Note:**

{special_note}

You are given a **candidate {node_name} set** for review. Your job is to:


1. Evaluate whether this candidate {node_name} set needs improvement by checking how well it adheres to the provided generation rules..

2. If improvement is needed, generate a revised and higher-quality version of the {node_name} set that better satisfies the rules and supports downstream LLM routing decisions.

Current Candidate {node_name} Set:

{candidate_node_set}

**Node Generation Rules:**

{node_gen_rules}

**Output Format:**
<justification>
Explain whether the current {node_name} set is flawed or could be improved. Mention overlap, vagueness, gaps, etc.
</justification>

<revision node begin>
{node_name_short}: node name
Definition: One-sentence definition  
Example: Example query or task

{node_name_short}: node name
Definition: One-sentence definition  
Example: Example query or task
</revision node end>

**Output:**"""

NodeSetChoicePrompt = AgenticiRouterKGSysPrompt + """
**Sepcial Note:**

{special_note}

Your Task: You are given **two candidate {node_name} sets**, labeled **Set A** and **Set B**. Your job is to:

1. Compare both sets based on how well they follow the generation rules.  
2. Select the better set — the one that provides more clarity, distinctiveness, usefulness, and alignment with routing goals.
3. Justify your choice in detail.

**Node Generation Rules:**

{node_gen_rules}

**Candidate Sets:**

Set A:

{candidate_node_set_a}

Set B:

{candidate_node_set_b}

**Output Format:**
<justification>
Explain why one set is better than the other. Reference the rules. Mention clarity, distinctiveness, coverage, usefulness, etc.
</justification>
<preferred set>
Set A / Set B
</preferred set>

**Output:**
"""

InitialDomains = [MathDomain, CreativeWritingDomain, CommonsenseKnowledgeDomain, ProgrammingDomain, LongFormQuestionAnsweringDomain, ReadingComprehensionDomain]
SuppDomains = [LegalDomain, MedicalQADomain]
NodeGenPrompt = [DomainNodePrompt, SubcategoryNodePrompt, DifficultyLevelNodePrompt, UserPreferenceNodePrompt]
NodeGenRule = [DomainNodeRule, SubcategoryNodeRule, DifficultyLevelNodeRule, UserPreferenceNodeRule]

NodeLevels = ["Domain Node", "Subcategory Node", "Difficulty Level Node", "User Preference Node"]
NodeNameShort = ["Domain", "Subcategory", "Level", "Preference"]







