import os
import sys
import json
import csv
import time
import logging
import re
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAGEvaluator")

try:
    from src.chains.rag_chain import RAGChain
    from src.utils.config import settings
except ImportError as e:
    logger.error(f"CRITICAL ERROR: Could not import project modules. Ensure you are running this from the project root. {e}")
    sys.exit(1)

@dataclass
class QueryRecord:
    query_id: str
    question: str
    gold_chunk_text: str
    source_doc: str

@dataclass
class EvalResult:
    query_id: str
    question: str
    gold_chunk_found: bool
    recall: float
    mrr: float
    ndcg: float
    context_precision: float
    latency_ms: float
    rank_found: int
    document: str
    page: str
    score: str
    chunk_ids: str
    retrieved_chunks: str

class MetricsCalculator:
    
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    @staticmethod
    def is_match(gold_text: str, retrieved_text: str) -> bool:
        norm_gold = MetricsCalculator.normalize_text(gold_text)
        norm_retrieved = MetricsCalculator.normalize_text(retrieved_text)
        return norm_gold in norm_retrieved

    @staticmethod
    def evaluate_results(gold_text: str, retrieved_chunks: List[Any], k: int = 10) -> Dict[str, Any]:
        
        metrics = {
            "gold_chunk_found": False,
            "rank_found": -1,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "context_precision": 0.0,
            "matching_chunk": None
        }

        if not retrieved_chunks:
            return metrics

        top_k_chunks = retrieved_chunks[:k]
        
        relevance_scores = []
        for rank, chunk in enumerate(top_k_chunks, start=1):
            if MetricsCalculator.is_match(gold_text, chunk.text):
                relevance_scores.append(1.0)
                if not metrics["gold_chunk_found"]:
                    metrics["gold_chunk_found"] = True
                    metrics["rank_found"] = rank
                    metrics["matching_chunk"] = chunk
            else:
                relevance_scores.append(0.0)
                
        #formula for recall, mrr, ndcg, and context precision
        metrics["recall"] = 1.0 if metrics["gold_chunk_found"] else 0.0

        metrics["mrr"] = (1.0 / metrics["rank_found"]) if metrics["gold_chunk_found"] else 0.0

        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
        
        idcg = 1.0 / math.log2(2) if metrics["gold_chunk_found"] else 1.0
        
        metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0
        total_relevant = sum(relevance_scores)
        metrics["context_precision"] = total_relevant / len(top_k_chunks) if top_k_chunks else 0.0

        return metrics

# Main evaluation class
class RAGEvaluator:
    def __init__(self):
        self.dataset_path = os.path.join(CURRENT_DIR, "eval_dataset.json")
        self.results_csv_path = os.path.join(CURRENT_DIR, "results.csv")
        self.summary_txt_path = os.path.join(CURRENT_DIR, "summary.txt")
        self.plots_dir = os.path.join(CURRENT_DIR, "plots")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.queries: List[QueryRecord] = []
        self.results: List[EvalResult] = []
        
        logger.info("Initializing RAGChain...")
        try:
            self.chain = RAGChain(settings)
            logger.info("RAGChain initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAGChain: {e}")
            sys.exit(1)

    def load_dataset(self):
        #loads and validates the dataset from a JSON file
        logger.info(f"Loading dataset from {self.dataset_path}")
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found at {self.dataset_path}")
            sys.exit(1)
            
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for item in data:
                if not all(k in item for k in ("query_id", "question", "gold_chunk_text", "source_doc")):
                    logger.warning(f"Skipping malformed item: {item}")
                    continue
                    
                self.queries.append(QueryRecord(
                    query_id=item["query_id"],
                    question=item["question"],
                    gold_chunk_text=item["gold_chunk_text"],
                    source_doc=item["source_doc"]
                ))
            logger.info(f"Successfully loaded {len(self.queries)} queries.")
        except Exception as e:
            logger.error(f"Failed to parse dataset JSON: {e}")
            sys.exit(1)

    def run_evaluations(self):
        logger.info("Starting evaluation pipeline...")
        start_runtime = time.time()
        
        for idx, query_record in enumerate(self.queries, start=1):
            logger.info(f"--- Processing Query {idx}/{len(self.queries)}: {query_record.query_id} ---")
            print(f"Current Query: {query_record.question}")
            query_start_time = time.time()
            try:
                retrieved_chunks = self.chain.retrieve_only(query_record.question)
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query_record.query_id}': {e}")
                retrieved_chunks = []
                
            latency_ms = (time.time() - query_start_time) * 1000
            
            # Grade Output
            metrics = MetricsCalculator.evaluate_results(query_record.gold_chunk_text, retrieved_chunks, k=10)
            
            # Extract metadata safely based on the project's RetrievedChunk structure
            matching_chunk = metrics["matching_chunk"]
            
            # Collate chunk IDs and full text for CSV
            chunk_ids_str = " | ".join([str(getattr(c, 'chunk_id', 'N/A')) for c in retrieved_chunks])
            retrieved_texts_str = "\n\n---\n\n".join([str(getattr(c, 'text', '')) for c in retrieved_chunks])
            
            result = EvalResult(
                query_id=query_record.query_id,
                question=query_record.question,
                gold_chunk_found=metrics["gold_chunk_found"],
                recall=metrics["recall"],
                mrr=metrics["mrr"],
                ndcg=metrics["ndcg"],
                context_precision=metrics["context_precision"],
                latency_ms=latency_ms,
                rank_found=metrics["rank_found"],
                document=getattr(matching_chunk, 'document', 'N/A') if matching_chunk else 'N/A',
                page=str(getattr(matching_chunk, 'page', 'N/A')) if matching_chunk else 'N/A',
                score=str(getattr(matching_chunk, 'score', 'N/A')) if matching_chunk else 'N/A',
                chunk_ids=chunk_ids_str,
                retrieved_chunks=retrieved_texts_str
            )
            
            self.results.append(result)
            
            #Output per query
            print(f"Recall: {result.recall:.2f} | MRR: {result.mrr:.2f} | nDCG: {result.ndcg:.2f} | Context Precision: {result.context_precision:.2f} | Latency: {result.latency_ms:.2f}ms")
            print(f"Retrieved {len(retrieved_chunks)} chunks.\n")
            
        self.total_runtime = time.time() - start_runtime
        logger.info(f"Evaluation complete in {self.total_runtime:.2f} seconds.")

    def save_csv(self):
        #Generates results.csv file
        logger.info(f"Saving results to {self.results_csv_path}")
        try:
            with open(self.results_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Write Headers
                headers = [
                    "query_id", "question", "gold_chunk_found", "recall", "mrr", 
                    "ndcg", "context_precision", "latency_ms", "rank_found", 
                    "document", "page", "score", "chunk_ids", "retrieved_chunks"
                ]
                writer.writerow(headers)
                
                # Write Data
                for r in self.results:
                    writer.writerow([
                        r.query_id, r.question, r.gold_chunk_found, r.recall, r.mrr,
                        r.ndcg, r.context_precision, r.latency_ms, r.rank_found,
                        r.document, r.page, r.score, r.chunk_ids, r.retrieved_chunks
                    ])
            print("CSV saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def save_summary(self):
        #Generates summary.txt file
        logger.info(f"Saving summary to {self.summary_txt_path}")
        
        num_queries = len(self.results)
        if num_queries == 0:
            logger.warning("No results to summarize.")
            return

        avg_recall = sum(r.recall for r in self.results) / num_queries
        avg_mrr = sum(r.mrr for r in self.results) / num_queries
        avg_ndcg = sum(r.ndcg for r in self.results) / num_queries
        avg_cp = sum(r.context_precision for r in self.results) / num_queries
        avg_latency = sum(r.latency_ms for r in self.results) / num_queries
        
        # Identify Best and Worst Queries based on MRR
        best_query = max(self.results, key=lambda x: x.mrr).query_id if any(r.mrr > 0 for r in self.results) else "None"
        worst_query = min(self.results, key=lambda x: x.mrr).query_id
        
        summary_text = f"""=================================================
Mini-RAG Retrieval Evaluation Summary
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Number of Queries: {num_queries}
Recall@10: {avg_recall:.4f}
MRR: {avg_mrr:.4f}
nDCG@10: {avg_ndcg:.4f}
Context Precision: {avg_cp:.4f}
Average Latency: {avg_latency:.2f} ms
Best Query: {best_query}
Worst Query: {worst_query}
Total Runtime: {self.total_runtime:.2f} seconds
=================================================
"""
        try:
            with open(self.summary_txt_path, mode="w", encoding="utf-8") as f:
                f.write(summary_text)
            print("Summary saved successfully.")
            print("\nFinal averages:")
            print(summary_text)
        except Exception as e:
            logger.error(f"Failed to save summary text: {e}")

    def generate_plots(self):
        logger.info(f"Generating plots in {self.plots_dir}")
        
        if not self.results:
            logger.warning("No results to plot.")
            return

        queries = [r.query_id for r in self.results]
        x_pos = np.arange(len(queries))

        def make_plot(data, title, ylabel, filename, color):
            plt.figure(figsize=(12, 6))
            plt.bar(x_pos, data, color=color, alpha=0.7)
            plt.plot(x_pos, data, color='black', marker='o', linestyle='-', linewidth=1, markersize=4)
            plt.title(title, fontsize=14)
            plt.xlabel("Query ID", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.xticks(x_pos, queries, rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            filepath = os.path.join(self.plots_dir, filename)
            try:
                plt.savefig(filepath)
                plt.close()
            except Exception as e:
                logger.error(f"Failed to save plot {filename}: {e}")

        
        make_plot([r.recall for r in self.results], "Recall@10 per Query", "Recall Score", "recall_plot.png", "skyblue")
        make_plot([r.mrr for r in self.results], "Mean Reciprocal Rank (MRR) per Query", "MRR Score", "mrr_plot.png", "lightgreen")
        make_plot([r.latency_ms for r in self.results], "Retrieval Latency per Query", "Latency (ms)", "latency_plot.png", "salmon")
        make_plot([r.ndcg for r in self.results], "nDCG@10 per Query", "nDCG Score", "ndcg_plot.png", "mediumpurple")
        make_plot([r.context_precision for r in self.results], "Context Precision per Query", "Precision Score", "context_precision_plot.png", "gold")
        
        print("Plots saved successfully.\n")

#Main execution
if __name__ == "__main__":
    logger.info("Initializing Evaluation Script...")
    evaluator = RAGEvaluator()
    evaluator.load_dataset()
    evaluator.run_evaluations()
    evaluator.save_csv()
    evaluator.save_summary()
    evaluator.generate_plots()
    
    logger.info("Person A evaluation tasks completed successfully. Data is ready for Person B.")