#--- START OF FILE supervisor_reasoning_pick_metric.py ---

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Dict
from autogen_core.models import UserMessage, ModelInfo
import re
from typing import List
from openai import AsyncOpenAI
import json
from openai import OpenAI
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from json_repair import repair_json
import os
import random 







SIMPLE_MATH_METRIC_1 = {
    "name": "CRITICAL_MATH_LOGIC_AUDIT",
    "detailed_definition": "A focused audit to detect substantive logical fallacies, calculation errors, or conditional oversights that invalidate the final result.",
    "evaluator_prompt": {
    
        "trigger_condition": "The Agent is performing mathematical reasoning, derivation, or calculation.",
        
   
        "risk_alert": (
            "You are an Objective Math Auditor. Your duty is to **verify** the agent's logic, not to rewrite their solution.\n"
            "**Audit Standards:**\n"
            "1. **Fatal Errors ONLY**: Flag specific steps that are mathematically FALSE. Do not critique efficiency, style, or 'better methods'. If the logic holds, let it pass.\n"
            "2. **Verify, Don't Assume**: Don't just look at the answer. Check if the intermediate deductions actually support the conclusion.\n\n"
            "**Potential Risk Areas to Scan (Heuristics):**\n"
            "- **Hallucinations**: Using non-existent theorems or making up numbers.\n"
            "- **Logic Gaps**: Jumping to conclusions without proof (e.g., assuming symmetry/maximums).\n"
            "- **Boundary Neglect**: Missing edge cases (zero, negative, empty sets) or necessary/sufficient conditions.\n"
            "- **Calculation Failures**: Basic arithmetic errors that propagate to the final result.\n"
            "3. Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
        )
    }
}

SIMPLE_CODE_METRIC_1 = {
    "name": "CRITICAL_CODE_CORRECTNESS_CHECK",
    "detailed_definition": "A functional audit focusing on runtime safety, logical integrity, and adherence to requirements in code implementation.",
    "evaluator_prompt": {
      
        "trigger_condition": "The Agent is generating, debugging, or analyzing computer code.",
        
    
        "risk_alert": (
            "You are a Senior Code Auditor. Your specific goal is to catch bugs that would cause the code to **crash** or produce **wrong outputs**.\n"
            "**Audit Standards:**\n"
            "1. **Functionality First**: Ignore PEP8, variable naming, or comments unless they obscure logic. Focus on: Will this code run? Will it output the right thing?\n"
            "2. **Logic over Syntax**: While syntax errors are bad, subtle logic bugs (that run but give wrong answers) are worse.\n\n"
            "**Potential Risk Areas to Scan (Heuristics):**\n"
            "- **Runtime Hazards**: Infinite loops, recursion without base cases, division by zero, index out of bounds.\n"
            "- **Library Misuse**: Hallucinated methods or incorrect API usage.\n"
            "- **Edge Cases**: Handling of empty inputs, 0, 1, or extreme values.\n"
            "- **Logic Bugs**: Off-by-one errors, incorrect conditional branches, or variable shadowing."
            "3. Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
        )
    }
}






METRIC_TEMPLATE_RANDOM_MATH = """
You are an Objective Logic Auditor.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL LOGIC ERROR** regarding a specific **Area of Concern**.

### 🛑 Relevance Pre-Check (CRITICAL)
Before auditing, you must strictly evaluate if the **[Area of Concern]** is actually relevant to the current Task and Agent Output.
- **If Irrelevant**: (e.g., the metric checks "Probability" but the task is "Geometry"), you must **STOP** and PASS the agent. In the JSON, write "Metric not applicable" in `analysis`, "N/A" in `suggestion`, and set `is_flawed` to `false`.
- **If Relevant**: Proceed to the Impact & Action Protocol below.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's reasoning is correct unless you find irrefutable evidence of a fatal flaw.
2. **The "Actionability" Test**: If you cannot provide a specific, mathematical correction (a formula, a step, or a value), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the Agent's phrasing is imperfect but the **FINAL ANSWER** remains mathematically correct, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic part. Write 'N/A' if valid or irrelevant.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on logic, not style. Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix it (e.g., 'Change x to y', 'Apply formula Z'). If no fix is needed or possible (or metric is irrelevant), write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does the FINAL ANSWER or core conclusion change? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""









METRIC_TEMPLATE_RANDOM_CODE = """
You are a Senior Code Auditor and Architect.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL CODING ERROR** regarding a specific **Area of Concern**.

### 🛑 Relevance Pre-Check (CRITICAL)
Before auditing, you must strictly evaluate if the **[Area of Concern]** is technically applicable to the current Code.
- **If Irrelevant**: (e.g., the metric checks "Database" but the code is "Sorting Array"), you must **STOP** and PASS the agent. In the JSON, write "Metric not applicable" in `analysis`, "N/A" in `suggestion`, and set `is_flawed` to `false`.
- **If Relevant**: Proceed to the Impact & Action Protocol below.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's code is functionally correct unless you find irrefutable evidence of a fatal flaw (syntax error, logic bug, or interface violation).
2. **The "Actionability" Test**: If you cannot provide a specific code correction (a line change, a logic fix, or a parameter adjustment), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the code is inefficient, verbose, or stylistically non-standard but **EXECUTES CORRECTLY** and returns the right result, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic code snippet. Write 'N/A' if valid or irrelevant.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on functional correctness (bugs/crashes), not style (PEP8/comments).Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix the code (e.g., 'Change index i to i+1', 'Import module X'). If no fix is needed (or metric is irrelevant), write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does it fix a runtime error, infinite loop, or incorrect output? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""


METRIC_TEMPLATE_MATH_AUDIT = """
You are an Objective Logic Auditor.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL LOGIC ERROR** regarding a specific **Area of Concern**.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's reasoning is correct unless you find irrefutable evidence of a fatal flaw.
2. **The "Actionability" Test**: If you cannot provide a specific, mathematical correction (a formula, a step, or a value), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the Agent's phrasing is imperfect but the **FINAL ANSWER** remains mathematically correct, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic part. Write 'N/A' if valid.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on logic, not style. Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix it (e.g., 'Change x to y', 'Apply formula Z'). If no fix is needed or possible, write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does the FINAL ANSWER or core conclusion change? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""

METRIC_TEMPLATE_CODE_AUDIT = """
You are a Senior Code Auditor and Architect.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL CODING ERROR** regarding a specific **Area of Concern**.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's code is functionally correct unless you find irrefutable evidence of a fatal flaw (syntax error, logic bug, or interface violation).
2. **The "Actionability" Test**: If you cannot provide a specific code correction (a line change, a logic fix, or a parameter adjustment), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the code is inefficient, verbose, or stylistically non-standard but **EXECUTES CORRECTLY** and returns the right result, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic code snippet. Write 'N/A' if valid.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on functional correctness (bugs/crashes), not style (PEP8/comments).Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix the code (e.g., 'Change index i to i+1', 'Import module X'). If no fix is needed, write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does it fix a runtime error, infinite loop, or incorrect output? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""


SUMMARY_TEMPLATE = """[Instruction]
Analyze the provided Task and Agent Output to extract key features for metric retrieval.
Strictly respond with a JSON object.

### Context
Task: {task}
Agent Output: {agent_output}

### Output JSON Format
{{
    "problem_scenario": ["Keyword 1", "Keyword 2"],
    "agent_action": ["Action Keyword 1", "Action Keyword 2"]
}}
"""

class Supervisor():
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        domain: str,  
        
 
        direct_k: int = 5,                 
        random_k: int = 0,                 
        use_simple_audit: int = 0,         
        
        sample_times: int = 3,
        pass_rate: float = 1,
        prune_flag: bool = True,
        
   
        metric_pool_file: str = "",
        embedding_cache_file: str = "",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_api_base: str = "",
        

        preloaded_metrics: List[Dict] = None,
        preloaded_embeddings: np.ndarray = None,
    ):
        self.domain = domain.lower()
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.scoreboard: Dict[str, Dict] = {}
        self.sample_times = sample_times
        self.prune_flag = prune_flag
        self.pass_rate = pass_rate
        self.reflection_records = []
        

        self.direct_k = direct_k
        self.random_k = random_k
        self.use_simple_audit = use_simple_audit
        
        if self.use_simple_audit > 0:
            print(f"[Supervisor] Mode: SIMPLE AUDIT (Fixed General Metric). RAG Retrieval is DISABLED.")
        
        # Embedding Client
        self.embedding_model = embedding_model
        self.embedding_client = OpenAI(
            api_key=embedding_api_key,
            base_url=embedding_api_base,
        )

        if preloaded_metrics is not None and preloaded_embeddings is not None:
            self.metrics = preloaded_metrics
            self.detailed_definitions_embeddings = preloaded_embeddings
            return 

  
        if metric_pool_file and os.path.exists(metric_pool_file):
            print(f"[Supervisor] Loading metrics text from: {metric_pool_file}")
            with open(metric_pool_file, "r", encoding='utf-8') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = []

        emb_map = {}
        if embedding_cache_file and os.path.exists(embedding_cache_file):
            print(f"[Supervisor] Loading embedding cache from: {embedding_cache_file}")
            try:
                with open(embedding_cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if "name" in record and "vector" in record:
                                emb_map[record["name"]] = record["vector"]
            except Exception as e:
                print(f"[Supervisor Warning] Failed to load embedding cache: {e}")
        
        vectors_list = []
        missing_texts = []
        missing_indices = []
        
        print("[Supervisor] Building vector index...")
        for idx, metric in enumerate(self.metrics):
            name = metric['name']
            if name in emb_map:
                vectors_list.append(np.array(emb_map[name], dtype=np.float32))
            else:
                vectors_list.append(None) 
                missing_indices.append(idx)
                trigger_text = metric.get('evaluator_prompt', {}).get('trigger_condition', metric['detailed_definition'])
                missing_texts.append(trigger_text)
        
        if missing_texts:
            print(f"[Supervisor] Calculating {len(missing_texts)} missing embeddings...")
            try:
                response = self.embedding_client.embeddings.create(model=self.embedding_model, input=missing_texts)
                for i, data_item in enumerate(response.data):
                    target_idx = missing_indices[i]
                    vectors_list[target_idx] = np.array(data_item.embedding, dtype=np.float32)
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to calculate embeddings: {e}")
                for idx in missing_indices:
                    if vectors_list[idx] is None: vectors_list[idx] = np.zeros(1024, dtype=np.float32)

        if vectors_list:
            self.detailed_definitions_embeddings = np.stack(vectors_list)
        else:
            self.detailed_definitions_embeddings = np.array([])
            
        print(f"[Supervisor] Index ready. Shape: {self.detailed_definitions_embeddings.shape}")

    def _safe_parse_json(self, raw_content: str, source_stage: str) -> Dict:
    
        try:
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            candidate_str = json_match.group(0) if json_match else raw_content
            parsed_data = repair_json(candidate_str, return_objects=True)
            if isinstance(parsed_data, list):
                if len(parsed_data) > 0: parsed_data = parsed_data[0]
                else: raise ValueError("Parsed JSON is an empty list.")
            if not isinstance(parsed_data, dict):
                raise ValueError(f"Parsed data is Type {type(parsed_data)}, NOT dict.")
            return parsed_data
        except Exception as e:
            print(f"\n[JSON Parse Error in {source_stage}]: {e}")
            raise e

    async def _match_metrics(self, task: str, output: str, metric_names=None) -> List[Dict]:
        if metric_names is not None:
            return [m for m in self.metrics if m['name'] in metric_names]

     
        if self.random_k > 0:
            print(f"[Supervisor] Mode: Random Selection (k={self.random_k})")
            k = min(self.random_k, len(self.metrics))
            return random.sample(self.metrics, k)

        summary_resp = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": SUMMARY_TEMPLATE.format(task=task, agent_output=output)}],
            temperature=0.0,
            max_tokens=1000,
        )
        raw_summary = summary_resp.choices[0].message.content.strip()
        try:
            json_match = re.search(r'\{.*\}', raw_summary, re.DOTALL)
            summary_data = json.loads(json_match.group(0))
            query_text = "Problem Scenario: " + ", ".join(summary_data.get("problem_scenario", [])) + \
                         ". Agent Action: " + ", ".join(summary_data.get("agent_action", []))
        except:
            query_text = raw_summary

        print(f"Generated Search Query: {query_text}")


        emb_resp = self.embedding_client.embeddings.create(model=self.embedding_model, input=[query_text])
        query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)
        similarities = np.dot(self.detailed_definitions_embeddings, query_emb)
        
 
        print(f"[Supervisor] Mode: Direct Search (Top-{self.direct_k}).")
        top_k_indices = np.argsort(similarities)[-self.direct_k:][::-1]
        return [self.metrics[idx] for idx in top_k_indices]
   
    async def _calc_score(self, task, message: TextMessage | dict, role: str = "Unknown", metrics=None) -> List[Dict]: 
        judgements = []
        message_content = message.content if isinstance(message, TextMessage) else message['content']
        
        if metrics is not None:
            matched_metrics = metrics
        else:
            matched_metrics = await self._match_metrics(task, message_content)
        
     
        if self.random_k > 0:
            if self.domain == "code": target_template = METRIC_TEMPLATE_RANDOM_CODE
            else: target_template = METRIC_TEMPLATE_RANDOM_MATH
        elif self.domain == "code":
            target_template = METRIC_TEMPLATE_CODE_AUDIT
        else:
            target_template = METRIC_TEMPLATE_MATH_AUDIT
        
        for metric in matched_metrics:
            m_eval = metric.get('evaluator_prompt', {})
            trigger = m_eval.get('trigger_condition', 'N/A')
            risk_alert = m_eval.get('risk_alert', '') 
            audit_context = f"Context: {trigger}\nSpecific Risk: {risk_alert}"
            
            prompt = target_template.format(
                task=task,
                role=role,   
                agent_output=message_content,
                trigger_condition=audit_context 
            )
            
           
            for attempt in range(5):
                try:
                    completion = await self._model_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1500
                    )
                    res_raw = completion.choices[0].message.content.strip()
                    finding = self._safe_parse_json(res_raw, source_stage=f"Audit-{metric['name']}")
                    
            
                    evidence = finding.get("evidence_quote", "N/A")
                    analysis = finding.get("analysis", "N/A")
                    suggestion = finding.get("suggestion", "N/A")
                    impact = finding.get("impact_assessment", "NO")
                    raw_flawed = finding.get("is_flawed", False)

            
                    is_suggestion_valid = str(suggestion).lower() not in ["n/a", "none", "no suggestion", "", "null"]
                    is_impact_significant = "yes" in str(impact).lower()
                    
                    if raw_flawed:
                        final_verdict_bool = is_suggestion_valid and is_impact_significant
                    else:
                        final_verdict_bool = False

                    verdict_str = 'flawed' if final_verdict_bool else 'correct'
                    
                    judgements.append({
                        'metric': metric['name'],
                        'verdict': verdict_str,
                        'evidence_quote': evidence,
                        'reasoning': analysis,         
                        'suggestion': suggestion,      
                        'impact': impact,              
                        'is_triggered': True 
                    })
                    break 

                except Exception as e:
                    if attempt == 4: 
                        print(f"[Audit Fail] Metric '{metric['name']}' failed 5 times.")
        
        return judgements
    
  
    def update_scoreboard_with_results(self, message: TextMessage, judgements: list[dict]):
        if not self.prune_flag:
            self.scoreboard[message.id] = {
                "message": message,
                "judgements": judgements, 
                "is_pruned": False        
            }
            return

        pass_cnt = 0
        for judge in judgements:
            if judge['verdict'].lower() == 'correct':
                pass_cnt += 1
        
        is_pruned = (pass_cnt / len(judgements)) < self.pass_rate if judgements else False

        self.scoreboard[message.id] = {
            "message": message,
            "judgements": judgements,
            "is_pruned": is_pruned
        }
        
    async def judge(self, task: str, message: TextMessage, attempt_num: int, role: str = "Assistant", previous_metrics=None, session_metrics=None):
        if not self.prune_flag:
            return True, [], None, None 

        print("\n" + "="*80)
        print(f"--- round: {attempt_num} | Agent: {message.source} (Role: {role}) ---")
        print("="*80)
        
        current_metrics = []

   
        if self.use_simple_audit:
            print("[Supervisor] Using FIXED General Metric (Simple Audit Mode).")
        
            if "code" in self.domain or "mbpp" in self.domain:
                current_metrics = [SIMPLE_CODE_METRIC_1] 
            else:
                current_metrics = [SIMPLE_MATH_METRIC_1]
            
        else:
            
            print("[Supervisor] Retrieving NEW metrics from Pool...")
            message_content = message.content if isinstance(message, TextMessage) else message['content']
            current_metrics = await self._match_metrics(task, message_content)


        judgements = await self._calc_score(task, message, role=role, metrics=current_metrics)
        

        print(" -> " + ", ".join([j['metric'] for j in judgements]) if judgements else " ")


        pass_cnt = 0
        feedback_lines = []
        
        for j in judgements:
            is_correct = (j['verdict'].lower() == 'correct')
            metric_name = j['metric']
            
            if is_correct: 
                pass_cnt += 1
                status = "✅ Correct"
                print(f" - {metric_name}: {status}")
            else:
                status = "❌ Flawed"
                print(f" - {metric_name}: {status}")
                
            
                print(f"   - : {j.get('evidence_quote', 'N/A')}")
                print(f"   - : {j.get('reasoning', 'N/A')}")
                print(f"   - : {j.get('suggestion', 'N/A')}")
                print(f"   - : {j.get('impact', 'N/A')}")
                
           
                reason = j.get('reasoning', 'N/A')
                short_reason = (reason[:1000] + '...') if len(reason) > 1000 else reason
                
                item = (
                    f"- [{metric_name}]: {j.get('suggestion', 'N/A')}\n"
                    f"  (Auditor's Note: {short_reason})"
                )
                feedback_lines.append(item)
        
        total_metrics = len(judgements)
        pass_flag = (pass_cnt / total_metrics) >= self.pass_rate if total_metrics > 0 else True 
        
        if not pass_flag and feedback_lines:
            feedback_body = "\n".join(feedback_lines)
            feedback = (
                f"An external auditor has reviewed your previous output (Attempt {attempt_num}) and flagged some potential issues. "
                "Please review the following suggestions critically:\n\n"
                f"{feedback_body}\n\n"
                "**Instruction**:\n"
                "1. If you agree with the advice, please refine your solution.\n"
                "2. **If you are confident your original logic is correct, you may ignore this advice.**\n"
                "3. Please output the corrected solution."
            )
        else:
            feedback = None
            
        print("\n[3]judge:")
        print(f" -> pass rate: {pass_cnt}/{total_metrics} | threshold: {self.pass_rate:.0%} | result: {'pass' if pass_flag else 'fail'}")
        
        if feedback:
            print("\n[4] feedback:")
            print(feedback)
            
        print("="*80 + "\n")
        
        return pass_flag, judgements, feedback, current_metrics
    
    def prune_info(self, all_messages: List[TextMessage]) -> List[TextMessage]:
        if not self.prune_flag: return all_messages
        pruned_messages = []
        for msg in all_messages:
            if msg.id in self.scoreboard:
                if not self.scoreboard[msg.id]['is_pruned']:
                    pruned_messages.append(msg)
                else:
                    # print(f"Info: Message ID {msg.id} filtered out by supervisor.")
                    pass
            else:
                if msg.source != "user":
                    pruned_messages.append(msg)
        return pruned_messages
    
    def get_messages_above_threshold(self) -> List[TextMessage]:
        if not self.prune_flag:
            return [entry["message"] for entry in self.scoreboard.values()]
        return [entry["message"] for entry in self.scoreboard.values() if not entry["is_pruned"]]
    
    def get_scores(self, role_map):
        ret_scores = {}
        for entry in self.scoreboard:
            ret_scores[entry] = {
                'role': role_map[self.scoreboard[entry]["message"].source],
                'message': self.scoreboard[entry]['message'].dump(),
                'judgements': self.scoreboard[entry]['judgements'],
                'is_pruned': self.scoreboard[entry]['is_pruned'],
            }
        return ret_scores
    
    def reset(self):
        self.scoreboard = {}