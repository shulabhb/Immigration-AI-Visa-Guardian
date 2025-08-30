#!/usr/bin/env python3
"""
Immigration Guardian RAG Chatbot
Uses FAISS vector search + Ollama LLM for immigration Q&A
"""

import json
import pathlib
import re
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

BASE = pathlib.Path(__file__).resolve().parents[1]
LAWS = BASE / "data" / "laws"

class ImmigrationRAGChatbot:
    def __init__(self, model_name: str = "llama3.2:latest"):
        """Initialize the RAG chatbot with FAISS indexes and Ollama LLM"""
        self.model_name = model_name
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Load all visa indexes
        self.indexes = {}
        self.metas = {}
        self.visa_types = ["F1", "F2", "H1B", "H4", "J1", "J2"]
        
        # Load visa-specific indexes
        for visa in self.visa_types:
            idx_path = LAWS / f"faiss_{visa}.index"
            meta_path = LAWS / f"faiss_{visa}_meta.json"
            
            if idx_path.exists() and meta_path.exists():
                self.indexes[visa] = faiss.read_index(str(idx_path))
                self.metas[visa] = json.load(open(meta_path, encoding="utf-8"))
                print(f"Loaded {visa} index: {len(self.metas[visa])} documents")
        
        # Load general index for fallback
        general_idx_path = LAWS / "faiss.index"
        general_meta_path = LAWS / "faiss_meta.json"
        
        if general_idx_path.exists() and general_meta_path.exists():
            self.indexes["general"] = faiss.read_index(str(general_idx_path))
            self.metas["general"] = json.load(open(general_meta_path, encoding="utf-8"))
            print(f"Loaded general index: {len(self.metas['general'])} documents")
        
        # Initialize knowledge base for common technical details
        self.knowledge_base = {
            "f1_opt_unemployment": {
                "standard_opt": "90 days total during the 12-month OPT period",
                "stem_opt": "150 days total (90 days from initial OPT + 60 days during STEM extension)",
                "source": "USCIS Policy Manual, Volume 2, Part F, Chapter 5"
            },
            "f1_cpt_limits": {
                "full_time_cpt": "12 months or more of full-time CPT eliminates OPT eligibility at the same educational level",
                "part_time_cpt": "Part-time CPT does not reduce OPT eligibility",
                "source": "8 CFR § 214.2(f)(10)(i)"
            },
            "h1b_cap": {
                "regular_cap": "65,000 visas per fiscal year",
                "masters_cap": "20,000 additional visas for advanced degree holders",
                "exemptions": "H-1B workers at institutions of higher education, nonprofit research organizations, and government research organizations are cap-exempt",
                "source": "INA § 214(g)"
            },
            "h4_ead_eligibility": {
                "requirements": "H-4 spouse must have H-1B spouse with approved I-140 or H-1B status extended beyond 6 years under AC21",
                "form": "Form I-765",
                "processing_time": "3-6 months",
                "source": "8 CFR § 274a.12(c)(26)"
            },
            "j1_waiver_categories": {
                "no_objection": "Home country government provides no-objection statement",
                "interested_government": "U.S. federal agency requests waiver",
                "persecution": "Fear of persecution based on race, religion, or political opinion",
                "exceptional_hardship": "Exceptional hardship to U.S. citizen or permanent resident spouse/child",
                "conrad_30": "Physicians working in underserved areas",
                "source": "INA § 212(e)"
            }
        }
    
    def classify_visa_type(self, query: str) -> str:
        """Simple rule-based visa classification with fuzzy matching for typos"""
        import re
        from difflib import SequenceMatcher
        
        query_lower = query.lower()
        
        # More precise greeting detection using word boundaries
        greeting_patterns = [
            r'\bhi\b', r'\bhello\b', r'\bhey\b', 
            r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b',
            r'\bhow are you\b', r'\bwhat\'s up\b'
        ]
        if any(re.search(pattern, query_lower) for pattern in greeting_patterns):
            return "general"
        
        # Common typos and abbreviations mapping
        typo_mapping = {
            "f.1": "f-1", "f.1": "f1", "f1": "f-1", "f1": "f1",
            "f.2": "f-2", "f.2": "f2", "f2": "f-2", "f2": "f2", 
            "hvb": "h-1b", "h1b": "h-1b", "h1b": "h1b",
            "h.1b": "h-1b", "h.1b": "h1b",
            "h.4": "h-4", "h4": "h-4", "h4": "h4",
            "j.1": "j-1", "j1": "j-1", "j1": "j1",
            "j.2": "j-2", "j2": "j-2", "j2": "j2"
        }
        
        # Check for potential typos in the query
        potential_typos = []
        for typo, correct in typo_mapping.items():
            if typo in query_lower:
                potential_typos.append((typo, correct))
        
        # Visa-specific keywords with weights
        visa_keywords = {
            "F1": ["f-1", "f1", "student", "study", "university", "college", "opt", "cpt", "on-campus", "off-campus", "practical training"],
            "F2": ["f-2", "f2", "dependent", "spouse", "child", "family"],
            "H1B": ["h-1b", "h1b", "work", "employment", "specialty", "occupation"],
            "H4": ["h-4", "h4", "dependent", "spouse", "child", "family"],
            "J1": ["j-1", "j1", "exchange", "visitor", "research", "scholar"],
            "J2": ["j-2", "j2", "dependent", "spouse", "child", "family"]
        }
        
        # Count keyword matches with better scoring
        scores = {}
        for visa, keywords in visa_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Give higher weight to exact visa mentions
                    if keyword in ["f-1", "f1", "f-2", "f2", "h-1b", "h1b", "h-4", "h4", "j-1", "j1", "j-2", "j2"]:
                        score += 3
                    else:
                        score += 1
            scores[visa] = score
        
        # Return visa with highest score, or "general" if no clear match
        best_visa = max(scores.items(), key=lambda x: x[1])
        
        # If we found potential typos and no clear visa match, suggest clarification
        if potential_typos and best_visa[1] == 0:
            # Create clarification message with more specific suggestions
            suggestions = []
            for typo, correct in potential_typos:
                if "f" in typo.lower():
                    if "1" in typo:
                        suggestions.append("F-1 (student visa)")
                    elif "2" in typo:
                        suggestions.append("F-2 (dependent visa)")
                    else:
                        suggestions.append("F-1 or F-2")
                elif "h" in typo.lower():
                    if "1" in typo or "b" in typo:
                        suggestions.append("H-1B (work visa)")
                    elif "4" in typo:
                        suggestions.append("H-4 (dependent visa)")
                    else:
                        suggestions.append("H-1B or H-4")
                elif "j" in typo.lower():
                    if "1" in typo:
                        suggestions.append("J-1 (exchange visitor)")
                    elif "2" in typo:
                        suggestions.append("J-2 (dependent visa)")
                    else:
                        suggestions.append("J-1 or J-2")
            
            if suggestions:
                return f"typo_clarification:{','.join(set(suggestions))}"
        
        return best_visa[0] if best_visa[1] > 0 else "general"
    
    def classify_question_type(self, query: str) -> str:
        """Classify the type of question to provide better responses"""
        query_lower = query.lower()
        
        # Technical/Detailed questions
        technical_keywords = [
            "exact", "specific", "precise", "limit", "requirement", "deadline", "timeline",
            "calculation", "formula", "percentage", "days", "hours", "weeks", "months",
            "unemployment", "cap", "quota", "prevailing wage", "lca", "i-765", "i-129",
            "sevis", "ds-2019", "i-20", "ead", "grace period", "extension"
        ]
        
        # Procedural questions
        procedural_keywords = [
            "how to", "step by step", "process", "procedure", "apply", "file", "submit",
            "application", "form", "document", "requirement", "checklist", "timeline",
            "deadline", "when to", "where to", "what forms", "which form"
        ]
        
        # Emergency/Urgent questions
        emergency_keywords = [
            "emergency", "urgent", "immediately", "right now", "today", "tomorrow",
            "expired", "expiring", "terminated", "laid off", "fired", "lost job",
            "out of status", "violation", "deportation", "removal", "overstay"
        ]
        
        # Comparison questions
        comparison_keywords = [
            "difference between", "vs", "versus", "compare", "similar", "different",
            "better", "worse", "advantage", "disadvantage", "pros", "cons"
        ]
        
        # Count matches
        technical_score = sum(1 for keyword in technical_keywords if keyword in query_lower)
        procedural_score = sum(1 for keyword in procedural_keywords if keyword in query_lower)
        emergency_score = sum(1 for keyword in emergency_keywords if keyword in query_lower)
        comparison_score = sum(1 for keyword in comparison_keywords if keyword in query_lower)
        
        # Return the highest scoring type
        scores = {
            "technical": technical_score,
            "procedural": procedural_score,
            "emergency": emergency_score,
            "comparison": comparison_score
        }
        
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else "general"
    
    def search_relevant_docs(self, query: str, visa_type: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents using FAISS with enhanced retrieval"""
        if visa_type not in self.indexes:
            return []
        
        # Encode query
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
        
        # Search FAISS index with more documents for complex questions
        question_type = self.classify_question_type(query)
        if question_type in ["technical", "procedural", "emergency"]:
            k = 8  # Get more sources for complex questions
        else:
            k = 5
        
        scores, indices = self.indexes[visa_type].search(query_vector, k)
        
        # Get documents with better filtering
        docs = []
        for i, score in zip(indices[0], scores[0]):
            if i >= 0 and score > 0.1:  # Filter out very low relevance scores
                doc = self.metas[visa_type][i].copy()
                doc['score'] = float(score)
                docs.append(doc)
        
        return docs
    
    def generate_answer(self, query: str, relevant_docs: List[Dict], visa_type: str = "general") -> str:
        """Generate answer using Ollama LLM with enhanced prompts"""
        if not relevant_docs:
            if visa_type == "general":
                return "Hello! I'm your Immigration Guardian. I can help you with questions about F-1, F-2, H-1B, H-4, J-1, and J-2 visa laws and regulations. What would you like to know?"
            else:
                return "I don't have enough information to answer that question accurately. Please try rephrasing or ask about a different immigration topic."
        
        # Classify question type for better prompting
        question_type = self.classify_question_type(query)
        
        # Create enhanced context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs[:5]):  # Use more documents for complex questions
            title = doc.get('title', 'Unknown')
            text = doc.get('text', '')[:800]  # Longer context for complex questions
            url = doc.get('url', '')
            section_hint = doc.get('section_hint', '')
            
            context_parts.append(f"Source {i+1}: {title}")
            if section_hint:
                context_parts.append(f"Section: {section_hint}")
            context_parts.append(f"Content: {text}")
            if url:
                context_parts.append(f"URL: {url}")
            context_parts.append("---")
        
        context = "\n".join(context_parts)
        
        # Inject knowledge base information for technical questions
        if question_type == "technical":
            context = self.inject_knowledge_base(query, context, visa_type)
        
        # Create specialized prompts based on question type
        if visa_type == "general":
            prompt = f"""You are an immigration law expert assistant. The user has asked a general greeting or question. Provide a brief, friendly welcome response that introduces your capabilities.

User Question: {query}

Please provide a short, welcoming response (2-3 sentences) that:
1. Acknowledges their greeting
2. Briefly mentions you can help with F-1, F-2, H-1B, H-4, J-1, and J-2 visa questions
3. Encourages them to ask a specific immigration question

Keep it concise and friendly."""
        
        elif question_type == "technical":
            prompt = f"""You are an immigration law expert assistant. The user has asked a technical question requiring specific details, numbers, or precise information.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Provide SPECIFIC numbers, dates, limits, and requirements when available
2. Cite exact regulatory sections (e.g., "8 CFR § 214.2(f)(10)(ii)")
3. Include specific form numbers and deadlines
4. If specific information is not in the context, clearly state what is known vs unknown
5. Provide actionable, concrete guidance
6. Use bullet points for clarity when listing requirements

Answer:"""
        
        elif question_type == "procedural":
            prompt = f"""You are an immigration law expert assistant. The user has asked a procedural question about how to do something.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Provide a STEP-BY-STEP process
2. Include specific form numbers and filing locations
3. Mention timelines and deadlines
4. List required documents and evidence
5. Include important tips and warnings
6. Use numbered steps for clarity
7. Mention any fees or costs involved

Answer:"""
        
        elif question_type == "emergency":
            prompt = f"""You are an immigration law expert assistant. The user has asked an urgent/emergency question.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Prioritize IMMEDIATE actions the user should take
2. Highlight any deadlines or time-sensitive requirements
3. Mention potential consequences if action is not taken
4. Provide contact information for USCIS or legal help if relevant
5. Be clear about what is urgent vs what can wait
6. Include any grace periods or extensions available
7. Advise consulting an immigration attorney for complex situations

Answer:"""
        
        elif question_type == "comparison":
            prompt = f"""You are an immigration law expert assistant. The user has asked a comparison question.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Create a clear comparison table or side-by-side analysis
2. Highlight key differences and similarities
3. Mention pros and cons of each option
4. Include specific requirements for each option
5. Provide recommendations based on common scenarios
6. Use clear formatting to distinguish between options

Answer:"""
        
        else:
            prompt = f"""You are an immigration law expert assistant. Answer the user's question based on the provided legal context.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Be accurate and helpful
2. Cite specific sources when possible
3. Provide practical guidance
4. Include relevant regulatory citations
5. Mention any important limitations or exceptions
6. Suggest next steps when appropriate

Answer:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower temperature for more factual responses
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I encountered an error generating the response.')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def inject_knowledge_base(self, query: str, context: str, visa_type: str) -> str:
        """Inject relevant knowledge base information into the context"""
        query_lower = query.lower()
        additional_info = []
        
        # Check for specific technical questions and inject relevant knowledge
        if visa_type == "F1":
            if any(term in query_lower for term in ["unemployment", "limit", "days", "opt"]):
                kb_info = self.knowledge_base.get("f1_opt_unemployment", {})
                if kb_info:
                    additional_info.append(f"OPT Unemployment Limits (Knowledge Base):")
                    additional_info.append(f"• Standard OPT: {kb_info.get('standard_opt', 'N/A')}")
                    additional_info.append(f"• STEM OPT: {kb_info.get('stem_opt', 'N/A')}")
                    additional_info.append(f"Source: {kb_info.get('source', 'N/A')}")
            
            if any(term in query_lower for term in ["cpt", "curricular", "practical training"]):
                kb_info = self.knowledge_base.get("f1_cpt_limits", {})
                if kb_info:
                    additional_info.append(f"CPT Limits (Knowledge Base):")
                    additional_info.append(f"• Full-time CPT: {kb_info.get('full_time_cpt', 'N/A')}")
                    additional_info.append(f"• Part-time CPT: {kb_info.get('part_time_cpt', 'N/A')}")
                    additional_info.append(f"Source: {kb_info.get('source', 'N/A')}")
        
        elif visa_type == "H1B":
            if any(term in query_lower for term in ["cap", "quota", "limit", "65,000", "20,000"]):
                kb_info = self.knowledge_base.get("h1b_cap", {})
                if kb_info:
                    additional_info.append(f"H-1B Cap Information (Knowledge Base):")
                    additional_info.append(f"• Regular Cap: {kb_info.get('regular_cap', 'N/A')}")
                    additional_info.append(f"• Masters Cap: {kb_info.get('masters_cap', 'N/A')}")
                    additional_info.append(f"• Exemptions: {kb_info.get('exemptions', 'N/A')}")
                    additional_info.append(f"Source: {kb_info.get('source', 'N/A')}")
        
        elif visa_type == "H4":
            if any(term in query_lower for term in ["work", "employment", "ead", "i-765"]):
                kb_info = self.knowledge_base.get("h4_ead_eligibility", {})
                if kb_info:
                    additional_info.append(f"H-4 EAD Eligibility (Knowledge Base):")
                    additional_info.append(f"• Requirements: {kb_info.get('requirements', 'N/A')}")
                    additional_info.append(f"• Form: {kb_info.get('form', 'N/A')}")
                    additional_info.append(f"• Processing Time: {kb_info.get('processing_time', 'N/A')}")
                    additional_info.append(f"Source: {kb_info.get('source', 'N/A')}")
        
        elif visa_type in ["J1", "J2"]:
            if any(term in query_lower for term in ["waiver", "2-year", "home residency", "212(e)"]):
                kb_info = self.knowledge_base.get("j1_waiver_categories", {})
                if kb_info:
                    additional_info.append(f"J-1 Waiver Categories (Knowledge Base):")
                    for category, description in kb_info.items():
                        if category != "source":
                            additional_info.append(f"• {category.replace('_', ' ').title()}: {description}")
                    additional_info.append(f"Source: {kb_info.get('source', 'N/A')}")
        
        if additional_info:
            return context + "\n\n" + "\n".join(additional_info)
        
        return context
    
    def chat(self, query: str) -> Dict:
        """Main chat function with enhanced response structure"""
        # Classify visa type
        visa_type = self.classify_visa_type(query)
        
        # Handle typo clarification
        if visa_type.startswith("typo_clarification:"):
            suggestions = visa_type.split(":")[1].split(",")
            clarification_msg = f"I noticed you might have a typo in your question. Did you mean one of these visa types?\n\n"
            for suggestion in suggestions:
                clarification_msg += f"• {suggestion}\n"
            clarification_msg += "\nPlease clarify which visa type you're asking about, and I'll be happy to help!"
            
            return {
                "query": query,
                "visa_type": "typo_clarification",
                "question_type": "clarification",
                "answer": clarification_msg,
                "sources": [],
                "num_sources": 0
            }
        
        # Classify question type
        question_type = self.classify_question_type(query)
        
        # Search for relevant documents
        relevant_docs = self.search_relevant_docs(query, visa_type)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_docs, visa_type)
        
        # Prepare sources for citation with enhanced metadata
        sources = []
        for doc in relevant_docs[:5]:  # Include more sources for complex questions
            sources.append({
                "title": doc.get("title", "Unknown"),
                "url": doc.get("url", ""),
                "section_hint": doc.get("section_hint", ""),
                "score": doc.get("score", 0),
                "visa_tags": doc.get("visa_tags", [])
            })
        
        return {
            "query": query,
            "visa_type": visa_type,
            "question_type": question_type,
            "answer": answer,
            "sources": sources,
            "num_sources": len(relevant_docs)
        }

def main():
    """Interactive chat interface"""
    print("🤖 Immigration Guardian RAG Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    # Initialize chatbot
    chatbot = ImmigrationRAGChatbot()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            print("🤔 Thinking...")
            
            # Get response
            response = chatbot.chat(query)
            
            print(f"\n🤖 Assistant: {response['answer']}")
            print(f"\n📋 Detected Visa Type: {response['visa_type']}")
            print(f"📚 Sources Used: {response['num_sources']}")
            
            if response['sources']:
                print("\n📖 Sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['title']}")
                    if source['url']:
                        print(f"     URL: {source['url']}")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
