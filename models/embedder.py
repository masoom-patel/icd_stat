"""
ICDEmbedder: Text encoder for ICD/HCC descriptions using medical embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config.model_config import get_config

logger = logging.getLogger(__name__)

class ICDEmbedder:
    """
    Text encoder for ICD/HCC descriptions using medical embeddings.
    Supports multiple embedding models with fallback options.
    """
    
    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the ICDEmbedder.
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory to cache embeddings
        """
        self.config = get_config()
        self.model_name = model_name or self.config.embedding_model_name
        self.cache_dir = Path(cache_dir or self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.embedding_dim = self.config.embedding_dim
        self.embeddings_cache = {}
        
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE and "sentence-transformers" in self.model_name:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_type = "sentence_transformers"
                
            elif TRANSFORMERS_AVAILABLE and any(model_prefix in self.model_name.lower() 
                                               for model_prefix in ["bert", "biobert", "clinical"]):
                logger.info(f"Loading Transformers model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model_type = "transformers"
                
            else:
                # Fallback to basic sentence transformers
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    logger.warning(f"Model {self.model_name} not available, falling back to MiniLM")
                    self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    self.model = SentenceTransformer(self.model_name)
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    self.model_type = "sentence_transformers"
                else:
                    raise ImportError("No suitable embedding libraries available")
                    
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            if self.config.fallback_embedding:
                self._create_fallback_embedder()
            else:
                raise
    
    def _create_fallback_embedder(self):
        """Create a simple TF-IDF based fallback embedder."""
        logger.warning("Using TF-IDF fallback embedder")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.model = TfidfVectorizer(max_features=384, stop_words='english')
        self.model_type = "tfidf"
        self.embedding_dim = 384
        self._tfidf_fitted = False
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        if self.model_type == "sentence_transformers":
            return self._encode_with_sentence_transformers(cleaned_texts, batch_size)
        elif self.model_type == "transformers":
            return self._encode_with_transformers(cleaned_texts, batch_size)
        elif self.model_type == "tfidf":
            return self._encode_with_tfidf(cleaned_texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _encode_with_sentence_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode texts using SentenceTransformers."""
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding with SentenceTransformers: {str(e)}")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def _encode_with_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode texts using Transformers library."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt')
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Use mean pooling of last hidden states
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(batch_embeddings.numpy())
                    
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size}: {str(e)}")
                # Fallback to random embeddings for this batch
                batch_size_actual = len(batch_texts)
                fallback_emb = np.random.randn(batch_size_actual, 768).astype(np.float32)
                embeddings.append(fallback_emb)
        
        return np.vstack(embeddings)
    
    def _encode_with_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF (fallback method)."""
        try:
            if not self._tfidf_fitted:
                self.model.fit(texts)
                self._tfidf_fitted = True
            
            tfidf_matrix = self.model.transform(texts)
            return tfidf_matrix.toarray().astype(np.float32)
        except Exception as e:
            logger.error(f"Error with TF-IDF encoding: {str(e)}")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for embedding."""
        if pd.isna(text) or text is None:
            return "unknown medical condition"
        
        text = str(text).strip().lower()
        if not text:
            return "unknown medical condition"
        
        # Basic cleaning
        text = text.replace('_', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def encode_icd_descriptions(self, icd_df: pd.DataFrame, 
                               icd_col: str = 'ICDCode', 
                               desc_col: str = 'Description') -> Dict[str, np.ndarray]:
        """
        Encode ICD descriptions and return mapping from ICD code to embedding.
        
        Args:
            icd_df: DataFrame containing ICD codes and descriptions
            icd_col: Column name for ICD codes
            desc_col: Column name for descriptions
            
        Returns:
            Dictionary mapping ICD codes to embeddings
        """
        cache_file = self.cache_dir / f"icd_embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(cached_embeddings)} cached ICD embeddings")
                return cached_embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {str(e)}")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(icd_df)} ICD codes")
        
        icd_codes = icd_df[icd_col].tolist()
        descriptions = icd_df[desc_col].fillna("unknown medical condition").tolist()
        
        # Create enhanced descriptions by combining code and description
        enhanced_descriptions = [
            f"ICD code {code}: {desc}" for code, desc in zip(icd_codes, descriptions)
        ]
        
        embeddings = self.encode_texts(enhanced_descriptions)
        
        # Create mapping
        icd_embeddings = dict(zip(icd_codes, embeddings))
        
        # Cache the results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(icd_embeddings, f)
            logger.info(f"Cached {len(icd_embeddings)} ICD embeddings")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {str(e)}")
        
        return icd_embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def get_embedding_stats(self) -> Dict[str, Union[str, int, float]]:
        """Get statistics about the embedding model."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'embedding_dimension': self.embedding_dim,
            'cache_dir': str(self.cache_dir),
            'cached_embeddings': len(self.embeddings_cache)
        }