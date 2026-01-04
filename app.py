"""
Decision Partner — AI for Life's Hard Choices
==============================================
A multi-agent system that helps you THINK, not thinks FOR you.

Philosophy:
- AI should be a thinking partner, not an oracle
- The best mentors ask better questions
- Clarity > Answers
- The pipeline is the product

Architecture:
- Clarifier Agent: Surfaces the REAL decision
- Explorer Agent: Expands options you haven't considered  
- Challenger Agent: Tests assumptions (devil's advocate)
- Synthesizer Agent: Compiles insights WITHOUT prescribing

Technologies: LangGraph, Hugging Face Inference Providers, Gradio
Author: Chad Sellers | linkedin.com/in/chadsellers
"""

import os
import time
import gradio as gr
from typing import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Hugging Face
from huggingface_hub import InferenceClient

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_inference_client():
    """Get HuggingFace InferenceClient with proper provider setup."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return None
    return InferenceClient(token=hf_token)

def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Call LLM using HuggingFace Inference Providers."""
    client = get_inference_client()
    
    if client is None:
        return "[Error: HF_TOKEN not set. Please add your Hugging Face token in Space Settings → Repository secrets.]"
    
    models_to_try = [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    
    for model in models_to_try:
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            if response and response.choices:
                return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = f"{model}: {str(e)}"
            continue
    
    return f"[Error connecting to inference providers. Last error: {last_error}]"

# ============================================================================
# STATE DEFINITION
# ============================================================================

class DecisionState(TypedDict):
    user_decision: str
    probing_questions: str
    clarified_decision: str
    hidden_options: str
    assumptions_challenged: str
    final_synthesis: str
    current_step: str

# ============================================================================
# AGENT PROMPTS
# ============================================================================

CLARIFIER_PROMPT = """You are the Clarifier — a thoughtful thinking partner who helps people discover what they're REALLY deciding.

Often, the decision someone presents isn't the actual decision. "Should I take this job?" might really be "Am I ready to leave the city I love?" or "Do I trust myself to handle change?"

Your job:
1. Read the user's decision carefully
2. Ask 2-3 probing questions that dig deeper (be specific to THEIR situation)
3. Reframe what the REAL decision might be

Be warm, curious, and non-judgmental. You're not here to solve — you're here to clarify.

USER'S DECISION:
{decision}

Respond in this exact format:

## Probing Questions

1. [First probing question specific to their situation]
2. [Second probing question that goes deeper]
3. [Third question that might reveal what's really at stake]

## The Real Decision Might Be...

[2-3 sentences reframing the decision to get at what's really at stake emotionally or personally]"""

EXPLORER_PROMPT = """You are the Explorer — a creative thinking partner who helps people see options they haven't considered.

When we're stuck in a decision, we often get tunnel vision. Two options feel like the only options. Your job is to gently expand the possibility space.

ORIGINAL DECISION:
{decision}

CLARIFIED DECISION:
{clarified}

Generate 3 alternative options or approaches they might not have considered. Be creative but grounded — these should feel like genuine possibilities.

Respond in this exact format:

## Options You Might Not Have Considered

**Option 1: [Creative Name]**
[2-3 sentences describing this option and why it might be worth considering]

**Option 2: [Creative Name]**
[2-3 sentences describing this option and why it might be worth considering]

**Option 3: [Creative Name]**
[2-3 sentences describing this option and why it might be worth considering]"""

CHALLENGER_PROMPT = """You are the Challenger — a respectful devil's advocate who helps people test their assumptions.

Good decisions require stress-testing. Your job is to push back gently — not to be difficult, but to strengthen their thinking.

ORIGINAL DECISION:
{decision}

CLARIFIED DECISION:
{clarified}

OPTIONS EXPLORED:
{options}

Identify 3 assumptions they might be making and challenge each one thoughtfully.

Respond in this exact format:

## Assumptions Worth Questioning

**Assumption 1: "[State the assumption]"**

*What if the opposite were true?* [2-3 sentences exploring the alternative perspective]

**Assumption 2: "[State the assumption]"**

*What if the opposite were true?* [2-3 sentences exploring the alternative perspective]

**Assumption 3: "[State the assumption]"**

*What if the opposite were true?* [2-3 sentences exploring the alternative perspective]"""

SYNTHESIZER_PROMPT = """You are the Synthesizer — a wise thinking partner who helps people see clearly without telling them what to do.

CRITICAL RULE: You must NEVER tell them what to decide. No "I think you should..." or "The best choice is..." Your job is to summarize and reflect, not prescribe.

ORIGINAL DECISION:
{decision}

CLARIFIED DECISION:
{clarified}

OPTIONS EXPLORED:
{options}

ASSUMPTIONS CHALLENGED:
{challenges}

Synthesize the key insights and end with a powerful reflective question.

Respond in this exact format:

## What's Become Clearer

[2-3 paragraphs synthesizing the key insights that emerged from this exploration. What patterns emerged? What tensions became visible? What new understanding surfaced?]

## The Heart of It

[One powerful sentence that captures the core tension, opportunity, or truth that emerged]

## A Question to Sit With

[One reflective question to help them move forward — NOT a recommendation. Make it personal and specific to their situation.]"""

# ============================================================================
# AGENT NODES
# ============================================================================

def clarifier_agent(state: DecisionState) -> dict:
    prompt = CLARIFIER_PROMPT.format(decision=state["user_decision"])
    response = call_llm(prompt)
    
    parts = response.split("## The Real Decision Might Be...")
    probing = parts[0].replace("## Probing Questions", "").strip() if parts else response
    clarified = parts[1].strip() if len(parts) > 1 else ""
    
    return {
        "probing_questions": probing,
        "clarified_decision": clarified,
        "current_step": "clarified"
    }

def explorer_agent(state: DecisionState) -> dict:
    prompt = EXPLORER_PROMPT.format(
        decision=state["user_decision"],
        clarified=state["clarified_decision"]
    )
    response = call_llm(prompt)
    return {"hidden_options": response, "current_step": "explored"}

def challenger_agent(state: DecisionState) -> dict:
    prompt = CHALLENGER_PROMPT.format(
        decision=state["user_decision"],
        clarified=state["clarified_decision"],
        options=state["hidden_options"]
    )
    response = call_llm(prompt)
    return {"assumptions_challenged": response, "current_step": "challenged"}

def synthesizer_agent(state: DecisionState) -> dict:
    prompt = SYNTHESIZER_PROMPT.format(
        decision=state["user_decision"],
        clarified=state["clarified_decision"],
        options=state["hidden_options"],
        challenges=state["assumptions_challenged"]
    )
    response = call_llm(prompt)
    
    # Build the complete journey output
    final_output = f"""
<div style="max-width: 800px; margin: 0 auto;">

# 🧭 Your Decision Partner Journey

---

## 📝 The Decision You Brought

> *"{state["user_decision"]}"*

---

## 🔍 Stage 1: Clarify

*The Clarifier helped surface what you might really be deciding...*

### Probing Questions
{state["probing_questions"]}

### The Deeper Decision
{state["clarified_decision"]}

---

## 🌎 Stage 2: Explore

*The Explorer expanded your possibility space...*

{state["hidden_options"]}

---

## ⚡ Stage 3: Challenge

*The Challenger stress-tested your assumptions...*

{state["assumptions_challenged"]}

---

## 🎯 Stage 4: Synthesize

*The Synthesizer compiled your insights — without telling you what to choose...*

{response}

---

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white; text-align: center; margin-top: 30px;">

### 💭 Remember

**This isn't advice. It's a mirror for your own thinking.**

The decision is yours. The clarity is yours too.

*Think clearer. Decide better. Move forward.*

</div>

---

<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">

**Built with [LangGraph](https://langchain-ai.github.io/langgraph/) Multi-Agent Orchestration**

Created by [Chad Sellers](https://linkedin.com/in/chadsellers) • M.S. AI/ML • Staff Engineer

*The pipeline is the product.*

</div>

</div>
"""
    
    return {"final_synthesis": final_output, "current_step": "complete"}

# ============================================================================
# WORKFLOW
# ============================================================================

def build_workflow():
    workflow = StateGraph(DecisionState)
    
    workflow.add_node("clarifier", clarifier_agent)
    workflow.add_node("explorer", explorer_agent)
    workflow.add_node("challenger", challenger_agent)
    workflow.add_node("synthesizer", synthesizer_agent)
    
    workflow.add_edge(START, "clarifier")
    workflow.add_edge("clarifier", "explorer")
    workflow.add_edge("explorer", "challenger")
    workflow.add_edge("challenger", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# ============================================================================
# MAIN EXECUTION WITH STREAMING STATUS
# ============================================================================

def think_through_decision(decision: str, progress=gr.Progress()) -> str:
    """Run the complete thinking partner workflow with progress updates."""
    
    if not decision or len(decision.strip()) < 10:
        return """
## 👋 Welcome to Decision Partner

Please share a decision you're wrestling with. The more context you provide, the deeper the insights.

**Good examples:**
- "I've been offered a promotion but it means relocating. I love my current city but the career growth is significant..."
- "I'm thinking about leaving my stable job to start a business. I have savings but also responsibilities..."
- "I need to decide whether to have a difficult conversation with someone close to me..."
"""
    
    progress(0, desc="🔍 Clarifier is examining your decision...")
    
    app = build_workflow()
    
    initial_state = {
        "user_decision": decision.strip(),
        "probing_questions": "",
        "clarified_decision": "",
        "hidden_options": "",
        "assumptions_challenged": "",
        "final_synthesis": "",
        "current_step": "starting"
    }
    
    try:
        # Run with progress simulation (actual progress tracking would require streaming)
        progress(0.15, desc="🔍 Clarifier: Surfacing the real decision...")
        time.sleep(0.5)
        
        progress(0.35, desc="🌎 Explorer: Finding hidden options...")
        time.sleep(0.5)
        
        progress(0.55, desc="⚡ Challenger: Testing assumptions...")
        time.sleep(0.5)
        
        progress(0.75, desc="🎯 Synthesizer: Compiling insights...")
        
        result = app.invoke(initial_state)
        
        progress(1.0, desc="✅ Complete!")
        
        return result.get("final_synthesis", "Something went wrong. Please try again.")
        
    except Exception as e:
        return f"""
## ⚠️ Something went wrong

**Error:** {str(e)}

**Troubleshooting:**
1. Ensure HF_TOKEN is set in Space Settings → Repository secrets
2. Your token needs "Make calls to Inference Providers" permission
3. Try again in a moment — the model may be warming up
"""

# ============================================================================
# CUSTOM CSS
# ============================================================================

CUSTOM_CSS = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

/* Card Styling */
.input-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e9ecef;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    padding: 12px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Output Styling */
.output-container {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e9ecef;
    min-height: 400px;
}

/* Agent Pills */
.agent-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #666;
    padding: 20px;
    margin-top: 40px;
    border-top: 1px solid #e9ecef;
}

/* Blockquote styling */
blockquote {
    border-left: 4px solid #667eea;
    padding-left: 16px;
    font-style: italic;
    color: #555;
    margin: 16px 0;
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
"""

# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    """Create the polished Decision Partner interface."""
    
    with gr.Blocks(
        title="Decision Partner | Think Clearer",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            font=gr.themes.GoogleFont("Inter")
        )
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 16px; color: white; text-align: center; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">🧭 Decision Partner</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;">AI That Helps You Think — Not Thinks For You</p>
        </div>
        """)
        
        # Value Proposition
        gr.Markdown("""
        <div style="text-align: center; max-width: 700px; margin: 0 auto 30px auto;">
        
        **The best mentors don't give you answers. They ask better questions.**
        
        Decision Partner is a multi-agent AI system that guides you through a structured thinking process. 
        It won't tell you what to choose — it will help you find your own clarity.
        
        </div>
        """)
        
        # How It Works - Agent Cards
        gr.HTML("""
        <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin-bottom: 30px;">
            <div style="background: #f0f4ff; padding: 16px 20px; border-radius: 12px; text-align: center; min-width: 140px;">
                <div style="font-size: 24px;">🔍</div>
                <div style="font-weight: 600; color: #4c51bf;">Clarifier</div>
                <div style="font-size: 12px; color: #666;">Finds the real decision</div>
            </div>
            <div style="color: #ccc; display: flex; align-items: center;">→</div>
            <div style="background: #f0fff4; padding: 16px 20px; border-radius: 12px; text-align: center; min-width: 140px;">
                <div style="font-size: 24px;">🌎</div>
                <div style="font-weight: 600; color: #276749;">Explorer</div>
                <div style="font-size: 12px; color: #666;">Expands your options</div>
            </div>
            <div style="color: #ccc; display: flex; align-items: center;">→</div>
            <div style="background: #fffaf0; padding: 16px 20px; border-radius: 12px; text-align: center; min-width: 140px;">
                <div style="font-size: 24px;">⚡</div>
                <div style="font-weight: 600; color: #c05621;">Challenger</div>
                <div style="font-size: 12px; color: #666;">Tests assumptions</div>
            </div>
            <div style="color: #ccc; display: flex; align-items: center;">→</div>
            <div style="background: #faf5ff; padding: 16px 20px; border-radius: 12px; text-align: center; min-width: 140px;">
                <div style="font-size: 24px;">🎯</div>
                <div style="font-weight: 600; color: #6b46c1;">Synthesizer</div>
                <div style="font-size: 12px; color: #666;">Compiles insights</div>
            </div>
        </div>
        """)
        
        # Main Interface
        with gr.Row():
            with gr.Column(scale=2):
                decision_input = gr.Textbox(
                    label="What decision are you wrestling with?",
                    placeholder="Share the decision and any context that feels relevant. The more you share, the deeper the insights.\n\nExample: I've been offered a senior role at a new company. It's more money and a bigger title, but I'd have to leave a team I've built over 3 years. Part of me feels ready for the next challenge, but another part wonders if I'm running toward something or away from something...",
                    lines=8,
                    max_lines=12
                )
                
                submit_btn = gr.Button(
                    "🧭 Help Me Think This Through",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-top: 16px;">
                
                **💡 Tips for deeper insights:**
                - Share the context and stakes
                - Include what's making it hard  
                - Mention what you've already considered
                
                *The 4 agents will take 60-90 seconds to work through your decision.*
                
                </div>
                """)
                
            with gr.Column(scale=3):
                output = gr.Markdown(
                    value="""
<div style="text-align: center; padding: 60px 20px; color: #666;">

### 👋 Ready when you are

Share a decision you're wrestling with, and four AI agents will help you think it through.

**This isn't about finding the "right" answer.**

It's about gaining clarity so *you* can decide with confidence.

---

*Decisions people have explored here:*

Career changes • Relationship crossroads • Big moves  
Business decisions • Life transitions • Hard conversations

</div>
                    """,
                    elem_classes=["output-container"]
                )
        
        submit_btn.click(
            fn=think_through_decision,
            inputs=[decision_input],
            outputs=[output],
            show_progress="full"
        )
        
        # Philosophy Section
        gr.HTML("""
        <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; margin-top: 40px;">
            <h3 style="text-align: center; margin-bottom: 20px;">💭 The Philosophy</h3>
            <div style="max-width: 600px; margin: 0 auto; text-align: center;">
                <blockquote style="font-size: 1.1em; font-style: italic; color: #555; border-left: 4px solid #667eea; padding-left: 20px; margin: 20px 0;">
                    "AI's highest value isn't in doing your thinking for you — it's in helping you think more clearly."
                </blockquote>
                <p style="color: #666;">
                    Most AI is designed to answer questions as fast as possible. 
                    But some questions shouldn't be answered quickly — they should be <em>explored</em>.
                </p>
                <p style="color: #667eea; font-weight: 600; margin-top: 16px;">
                    The pipeline is the product.
                </p>
            </div>
        </div>
        """)
        
        # Technical Footer
        gr.HTML("""
        <div style="text-align: center; padding: 30px; margin-top: 40px; border-top: 1px solid #e9ecef;">
            <p style="color: #666; margin-bottom: 12px;">
                <strong>Built with</strong> 
                <a href="https://langchain-ai.github.io/langgraph/" target="_blank" style="color: #667eea;">LangGraph</a> Multi-Agent Orchestration • 
                <a href="https://huggingface.co/docs/inference-providers" target="_blank" style="color: #667eea;">HF Inference Providers</a> • 
                <a href="https://gradio.app" target="_blank" style="color: #667eea;">Gradio</a>
            </p>
            <p style="color: #888; font-size: 0.9em;">
                Created by <a href="https://linkedin.com/in/chadsellers" target="_blank" style="color: #667eea;">Chad Sellers</a> • 
                M.S. AI/ML, Colorado State University Global • 
                <a href="https://github.com/chadsellers-24-7-365" target="_blank" style="color: #667eea;">GitHub</a>
            </p>
            <p style="color: #667eea; font-weight: 600; margin-top: 16px;">
                Think clearer. Decide better. Move forward.
            </p>
        </div>
        """)
    
    return demo

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(ssr_mode=False)
