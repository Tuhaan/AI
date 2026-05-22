"""
Recommendation Engine
Uses TF-IDF vectorization and cosine similarity
to match user skills/interests against space content.
"""

from typing import List, Tuple, Dict, Any

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    def __init__(self, mongo_uri: str, db_name: str = "torchbearer"):
        """
        Initialize MongoDB connection.
        """

        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable is not set.")

        self.client = MongoClient(mongo_uri) #creates connection to mongodb
        self.db = self.client[db_name] #selects project database
        self.spaces_collection = self.db["spaces"] # targets spaces collection

        # Verify MongoDB connection or checks whether mongodb is alive
        try:
            self.client.admin.command("ping")
            print("✅ MongoDB connected successfully")

        except ConnectionFailure as exc:
            raise ConnectionError(
                f"❌ Cannot connect to MongoDB: {exc}"
            ) from exc

    def _fetch_spaces(self) -> List[Dict[str, Any]]: #Fetch all spaces from MongoDB. Expected fields: - _id,  - title - description- tags - createdBy
    

        pipeline = [
            {
                "$lookup": { #used for join operation(join spaces collection with users collection)
                    "from": "users",
                    "localField": "createdBy",
                    "foreignField": "_id",
                    "as": "creator",
                }
            },
            {
                "$project": { #used to return only necessary fields
                    "_id": 1,
                    "title": 1,
                    "name": 1,
                    "description": 1,
                    # Use $ifNull so spaces without tags field return []
                    "tags": {"$ifNull": ["$tags", []]},
                    "creator": {
                        "$arrayElemAt": ["$creator.name", 0]
                    },
                }
            },
        ]

        return list(self.spaces_collection.aggregate(pipeline))
    # to convert structured space data into searchable text
    def _build_searchable_text(
        self,
        space: Dict[str, Any]
    ) -> str:
        #Build searchable text from: - title - description - tags

        parts = []

        # Title
        title = (
            space.get("title")
            or space.get("name")
            or ""
        )

        if title:
            normalized_title = title.lower()

            # Weight title heavily to increase the importance of title terms
            parts.extend([normalized_title] * 5)

        # Description
        description = space.get("description", "")

        if description:
            parts.append(description.lower())

        # Tags
        tags = space.get("tags", [])

        normalized_tags = []

        for tag in tags:
            tag_lower = tag.lower()

            normalized_tags.append(tag_lower)
            normalized_tags.append(
                tag_lower.replace(" ", "")
            )
            normalized_tags.append(
                tag_lower.replace("/", " ")
            )

        if normalized_tags:
            tag_text = " ".join(normalized_tags)

            # Weight tags heavily
            parts.extend([tag_text] * 4)

        return " ".join(parts)

    def recommend(
        self,
        query_terms: List[str],
        top_n: int = 5,
    ) -> Tuple[List[Dict[str, Any]], int]:

        spaces = self._fetch_spaces()

        if not spaces:
            return [], 0

        # Build corpus
        corpus = [
            self._build_searchable_text(space)
            for space in spaces
        ]

        # Process query terms
        processed_terms = []

        for term in query_terms:
            term_lower = term.lower()

            processed_terms.append(term_lower)
            processed_terms.append(
                term_lower.replace(" ", "")
            )
            processed_terms.append(
                term_lower.replace("/", " ")
            )

        query_text = " ".join(processed_terms)

        # Add query to corpus
        all_documents = corpus + [query_text]

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",  #remove common words like the, is , and
            ngram_range=(1, 2), #for improving semantic matching by using two-words phrases for single words like machine learning becomes machine and learning
            min_df=1,
            sublinear_tf=True, #applies logarithmic scaling by preventing keyword stuffing 1+log(tf)
        )

        # Transform documents
        tfidf_matrix = vectorizer.fit_transform(
            all_documents
        )

        # Separate vectors
        space_vectors = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]

        # Similarity calculation between user query vectors and space vectors
        similarity_scores = cosine_similarity(
            query_vector,
            space_vectors
        ).flatten() #calculates similarity between user interests and every space

        # Rank results by sorting highest similarity 1st
        ranked_indices = np.argsort(
            similarity_scores
        )[::-1]

        results = []

        for idx in ranked_indices[:top_n]:
            score = float(similarity_scores[idx])

            # Ignore very weak/irrelevant matches
            if score <= 0.01:
                break

            space = spaces[idx]

            results.append(
                {
                    "id": str(space["_id"]),
                    "title": (
                        space.get("title")
                        or space.get("name")
                        or "Untitled"
                    ),
                    "description": space.get("description") or "",
                    "tags": space.get("tags") or [],
                    "similarity_score": round(score, 4),
                    "created_by": space.get("creator") or None,
                }
            )

        return results, len(spaces)