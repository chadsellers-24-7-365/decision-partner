"""
Decision Partner — AI for Life's Hard Choices
==============================================
A multi-agent system that helps you THINK, not thinks FOR you.

Design: Minimalist black/white, clean typography, sharp edges
Architecture: Clarifier → Explorer → Challenger → Synthesizer

Author: Chad Sellers | linkedin.com/in/chadsellers
"""

import os
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from huggingface_hub import InferenceClient

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

def get_client():
    token = os.getenv("HF_TOKEN")
    return InferenceClient(token=token) if token else None

def call_llm(prompt: str) -> str:
    client = get_client()
    if not client:
        return "[Error: HF_TOKEN not set. Add it in Settings → Secrets.]"
    
    models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-72B-Instruct",
    ]
    
    for model in models:
        try:
            response = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            if response and response.choices:
                return response.choices[0].message.content.strip()
        except:
            continue
    
    return "[Error: Could not connect. Please try again.]"

# ============================================================================
# STATE
# ============================================================================

class DecisionState(TypedDict):
    decision: str
    clarified: str
    options: str
    challenges: str
    synthesis: str

# ============================================================================
# PROMPTS
# ============================================================================

CLARIFIER_PROMPT = """You help people discover what they're REALLY deciding.

The surface decision often hides a deeper one. "Should I take this job?" might really be "Do I trust myself to handle change?"

USER'S DECISION: {decision}

Respond with:

PROBING QUESTIONS:
1. [First question that digs deeper]
2. [Second question about what's really at stake]
3. [Third question about underlying fears or hopes]

THE REAL DECISION:
[2-3 sentences reframing what they're actually deciding]"""

EXPLORER_PROMPT = """You help people see options they haven't considered.

ORIGINAL: {decision}
REFRAMED: {clarified}

Generate 3 alternatives they might have missed. Be creative but realistic.

Respond with:

OPTION 1: [NAME]
[2 sentences on what this looks like and why it might work]

OPTION 2: [NAME]
[2 sentences on what this looks like and why it might work]

OPTION 3: [NAME]
[2 sentences on what this looks like and why it might work]"""

CHALLENGER_PROMPT = """You test assumptions respectfully. You're a thinking partner, not a critic.

DECISION: {decision}
REFRAMED: {clarified}
OPTIONS: {options}

Identify 3 assumptions and challenge each.

Respond with:

ASSUMPTION 1: "[What they assume]"
What if the opposite were true? [Explore this]

ASSUMPTION 2: "[What they assume]"
What if the opposite were true? [Explore this]

ASSUMPTION 3: "[What they assume]"
What if the opposite were true? [Explore this]"""

SYNTHESIZER_PROMPT = """You compile insights WITHOUT telling them what to choose.

CRITICAL: Never say "you should" or "I recommend." End with a question, not advice.

DECISION: {decision}
REFRAMED: {clarified}
OPTIONS: {options}
CHALLENGES: {challenges}

Respond with:

WHAT'S CLEARER NOW:
[2-3 paragraphs synthesizing the key insights]

THE CORE TENSION:
[One sentence capturing what this really comes down to]

A QUESTION TO SIT WITH:
[One powerful question to help them move forward]"""

# ============================================================================
# AGENTS
# ============================================================================

def clarifier(state: DecisionState) -> dict:
    response = call_llm(CLARIFIER_PROMPT.format(decision=state["decision"]))
    return {"clarified": response}

def explorer(state: DecisionState) -> dict:
    response = call_llm(EXPLORER_PROMPT.format(
        decision=state["decision"], 
        clarified=state["clarified"]
    ))
    return {"options": response}

def challenger(state: DecisionState) -> dict:
    response = call_llm(CHALLENGER_PROMPT.format(
        decision=state["decision"],
        clarified=state["clarified"],
        options=state["options"]
    ))
    return {"challenges": response}

def synthesizer(state: DecisionState) -> dict:
    response = call_llm(SYNTHESIZER_PROMPT.format(
        decision=state["decision"],
        clarified=state["clarified"],
        options=state["options"],
        challenges=state["challenges"]
    ))
    return {"synthesis": response}

# ============================================================================
# WORKFLOW
# ============================================================================

def build_workflow():
    wf = StateGraph(DecisionState)
    wf.add_node("clarifier", clarifier)
    wf.add_node("explorer", explorer)
    wf.add_node("challenger", challenger)
    wf.add_node("synthesizer", synthesizer)
    wf.add_edge(START, "clarifier")
    wf.add_edge("clarifier", "explorer")
    wf.add_edge("explorer", "challenger")
    wf.add_edge("challenger", "synthesizer")
    wf.add_edge("synthesizer", END)
    return wf.compile()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def think(decision: str, progress=gr.Progress()) -> str:
    if not decision or len(decision.strip()) < 10:
        return """
### Enter a decision above

Share something you're wrestling with. Career, relationships, life transitions — whatever's on your mind.

The more context you give, the better the agents can help.
"""
    
    progress(0.2, desc="Clarifying...")
    progress(0.4, desc="Exploring options...")
    progress(0.6, desc="Challenging assumptions...")
    progress(0.8, desc="Synthesizing...")
    
    app = build_workflow()
    result = app.invoke({
        "decision": decision.strip(),
        "clarified": "",
        "options": "",
        "challenges": "",
        "synthesis": ""
    })
    
    progress(1.0, desc="Done")
    
    output = f'''
---

## Your Decision

> {decision}

---

## 01 — Clarify

{result["clarified"]}

---

## 02 — Explore

{result["options"]}

---

## 03 — Challenge

{result["challenges"]}

---

## 04 — Synthesize

{result["synthesis"]}

---

<p style="text-align: center; color: #666; margin-top: 40px;">
<em>This isn't advice. It's a mirror for your thinking.</em><br>
The decision is yours.
</p>
'''
    return output

# ============================================================================
# MINIMAL CSS
# ============================================================================

CSS = """
/* Reset and base */
* { box-sizing: border-box; }

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Header */
.header {
    text-align: center;
    padding: 60px 20px 40px 20px;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 40px;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 12px 0;
    color: #000;
}

.header p {
    font-size: 1.1rem;
    color: #666;
    margin: 0;
    font-weight: 400;
}

/* Pipeline visualization */
.pipeline {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    padding: 30px 0;
    font-size: 0.85rem;
    color: #999;
}

.pipeline .step {
    padding: 8px 16px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    font-weight: 500;
    color: #333;
    background: #fafafa;
}

.pipeline .arrow {
    color: #ccc;
}

/* Input area */
.input-section {
    margin-bottom: 40px;
}

textarea {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: 16px !important;
    background: #fafafa !important;
    transition: border-color 0.2s, background 0.2s !important;
}

textarea:focus {
    border-color: #000 !important;
    background: #fff !important;
    outline: none !important;
}

/* Button */
button.primary {
    background: #000 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 32px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}

button.primary:hover {
    background: #333 !important;
}

/* Output */
.output-section {
    border-top: 1px solid #e0e0e0;
    padding-top: 40px;
    margin-top: 20px;
}

/* Typography in output */
.prose h2 {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #000;
    margin: 40px 0 20px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #000;
}

.prose blockquote {
    border-left: 2px solid #000;
    padding-left: 20px;
    margin: 20px 0;
    font-style: italic;
    color: #333;
}

.prose p {
    line-height: 1.7;
    color: #333;
}

.prose strong {
    font-weight: 600;
    color: #000;
}

/* Footer */
.footer {
    text-align: center;
    padding: 40px 20px;
    border-top: 1px solid #e0e0e0;
    margin-top: 60px;
    color: #999;
    font-size: 0.85rem;
}

.footer a {
    color: #666;
    text-decoration: none;
    border-bottom: 1px solid #ccc;
}

.footer a:hover {
    color: #000;
    border-color: #000;
}

/* Hide Gradio branding */
footer { display: none !important; }
"""

# ============================================================================
# UI
# ============================================================================

def create_ui():
    with gr.Blocks(css=CSS, title="Decision Partner") as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>Decision Partner</h1>
            <p>AI that helps you think — not thinks for you</p>
        </div>
        """)
        
        # Pipeline
        gr.HTML("""
        <div class="pipeline">
            <span class="step">Clarify</span>
            <span class="arrow">→</span>
            <span class="step">Explore</span>
            <span class="arrow">→</span>
            <span class="step">Challenge</span>
            <span class="arrow">→</span>
            <span class="step">Synthesize</span>
        </div>
        """)
        
        # Input
        with gr.Column(elem_classes="input-section"):
            decision_input = gr.Textbox(
                label="",
                placeholder="What decision are you wrestling with? Share the context — the more detail, the better the thinking.",
                lines=5,
                show_label=False
            )
            submit_btn = gr.Button("Think it through", variant="primary")
        
        # Output
        with gr.Column(elem_classes="output-section"):
            output = gr.Markdown(
                value="""
### How it works

Four AI agents will help you think through your decision:

1. **Clarify** — Surface what you're really deciding
2. **Explore** — Find options you haven't considered  
3. **Challenge** — Test your assumptions
4. **Synthesize** — Compile insights (without telling you what to do)

The process takes about 60 seconds.

---

*Career changes. Relationship crossroads. Life transitions. Business decisions.*

*Whatever's weighing on you.*
                """,
                elem_classes="prose"
            )
        
        submit_btn.click(
            fn=think,
            inputs=[decision_input],
            outputs=[output],
            show_progress="minimal"
        )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            Built with <a href="https://langchain-ai.github.io/langgraph/" target="_blank">LangGraph</a> · 
            <a href="https://github.com/chadsellers-24-7-365/decision-partner" target="_blank">View source</a> · 
            <a href="https://linkedin.com/in/chadsellers" target="_blank">Chad Sellers</a>
            <br><br>
            <em>Think clearer. Decide better. Move forward.</em>
        </div>
        """)
    
    return demo

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(ssr_mode=False)
