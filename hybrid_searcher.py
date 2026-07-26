from qdrant_client import QdrantClient, models


class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(url = 'http://localhost:6335')  # assumes Qdrant is running on default localhost:6334

    def search(self, text: str):
        dense_vector_name = "dense_vector"
        sparse_vector_name = "sparse_vector"

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=text, model=self.DENSE_MODEL),
                    using=dense_vector_name,
                ),
                models.Prefetch(
                    query=models.Document(text=text, model=self.SPARSE_MODEL),
                    using=sparse_vector_name,
                ),
            ],
            query_filter=None,
            limit=5,
        ).points

        return [point.payload for point in search_result]

    def search_with_city_filter(self, text: str, city_of_interest: str):
        city_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(value=city_of_interest),
                )
            ]
        )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=text, model=self.DENSE_MODEL),
            query_filter=city_filter,
            limit=5,
        ).points

        return [point.payload for point in search_result]


