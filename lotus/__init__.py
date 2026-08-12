import logging
import lotus.dtype_extensions
import lotus.models
import lotus.vector_store
import lotus.nl_expression
import lotus.templates
import lotus.utils
import lotus.ast
from lotus.sem_ops import (
    load_sem_index,
    sem_agg,
    sem_extract,
    sem_filter,
    sem_index,
    sem_join,
    sem_map,
    sem_partition_by,
    sem_search,
    sem_sim_join,
    sem_cluster_by,
    sem_group_by,
    sem_dedup,
    sem_topk,
)
from lotus.evals import llm_as_judge, pairwise_judge
from lotus.web_search import web_search, web_extract, WebSearchCorpus
from lotus.settings import settings  # type: ignore[attr-defined]
from lotus.ast import LazyFrame
from lotus.corpus import Corpus

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "sem_map",
    "sem_filter",
    "sem_agg",
    "sem_extract",
    "sem_join",
    "sem_partition_by",
    "sem_topk",
    "sem_index",
    "load_sem_index",
    "sem_sim_join",
    "sem_cluster_by",
    "sem_group_by",
    "sem_search",
    "sem_dedup",
    "settings",
    "nl_expression",
    "templates",
    "logger",
    "models",
    "vector_store",
    "utils",
    "dtype_extensions",
    "web_search",
    "web_extract",
    "WebSearchCorpus",
    "llm_as_judge",
    "pairwise_judge",
    "ast",
    "LazyFrame",
    "Optimizer",
    "Corpus",
]
