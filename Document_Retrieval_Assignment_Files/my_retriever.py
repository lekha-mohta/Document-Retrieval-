import math

class Retrieve:
    def __init__(self, index, term_weighting):
        """
        Initialize the Retrieve object with the given index and term weighting scheme.
        """
        self.index = index 
        self.term_weighting = term_weighting
        self.doc_ids = self.compute_number_of_documents() 
        self.num_of_docs = len(self.doc_ids) 
        self.idf = self.compute_idf()
        self.doc_vectors, self.doc_magnitudes = self.compute_document_vectors()

    def compute_number_of_documents(self):
        """
        Extract unique set of document IDs from the index.
        """
        doc_ids = set()
        for term in self.index:
            doc_ids.update(self.index[term])
        return doc_ids
    
    def compute_idf(self):
        """
        Compute the inverse document frequency (IDF) for each term in the index.
        """
        idf = {}
        for term in self.index:
            doc_freqeuncy = len(self.index[term]) 
            idf[term] = math.log(self.num_of_docs / doc_freqeuncy) if doc_freqeuncy else 0
        return idf

    def compute_document_vectors(self):
        """
        Generate document vectors and their magnitudes based on the selected term weighting scheme.
        """
        doc_vectors = {doc_id: {} for doc_id in self.doc_ids}
        doc_magnitudes = {}
        for term, postings in self.index.items():
            for doc_id, count in postings.items():
                if self.term_weighting == 'binary':
                    weight = 1 
                elif self.term_weighting == 'tf':
                    weight = count
                elif self.term_weighting == 'tfidf':
                    weight = count * self.idf.get(term, 0)
                
                doc_vectors[doc_id][term] = doc_vectors[doc_id].get(term, 0) + weight
                doc_magnitudes[doc_id] = doc_magnitudes.get(doc_id, 0) + weight**2

        for doc_id in doc_magnitudes:
            doc_magnitudes[doc_id] = math.sqrt(doc_magnitudes[doc_id])
        return doc_vectors, doc_magnitudes

    def compute_query_vector(self, query):
        """
        Compute the query vector based on the selected term weighting scheme.
        """
        query_vector = {}
        for term in set(query):
            if term in self.index:
                query_tf = query.count(term)
                if self.term_weighting == 'binary':
                    query_weight = 1
                elif self.term_weighting == 'tf':
                    query_weight = query_tf
                elif self.term_weighting == 'tfidf':
                    query_weight = query_tf * self.idf.get(term, 0)
                query_vector[term] = query_weight
        return query_vector

    def compute_query_magnitude(self, query_vector):
        """
        Compute the magnitude of the query vector.
        """
        return math.sqrt(sum(weight**2 for weight in query_vector.values()))

    def compute_cosine_similarity(self, dot_product, query_magnitude, doc_magnitude):
        """
        Compute the cosine similarity between the query and a document.
        """
        if query_magnitude != 0 and doc_magnitude != 0:
            return dot_product / (query_magnitude * doc_magnitude)
        return 0

    def for_query(self, query):
        """
        Retrieve the top 10 documents for the given query based on cosine similarity.
        """
        query_vector = self.compute_query_vector(query)
        query_magnitude = self.compute_query_magnitude(query_vector)
        
        doc_scores = {}
        for doc_id, doc_vector in self.doc_vectors.items():
            dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)
            cosine_similarity = self.compute_cosine_similarity(dot_product, query_magnitude, self.doc_magnitudes[doc_id])
            doc_scores[doc_id] = cosine_similarity

         # Applied threshold filtering (Score > 0.1) to remove low scores
        filtered_docs = {doc_id: score for doc_id, score in doc_scores.items() if score > 0.1}
        
        # Ranked the documents in descending order based on the cosine similarity score
        ranked_docs = sorted(filtered_docs.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, score in ranked_docs[:10]]

    def evaluate_metrics(self, query, relevant_docs):
        """
        Evaluate precision, recall, and F-measure for the given query and relevant documents.
        """
        retrieved_docs = self.for_query(query)
        true_positives = len(set(retrieved_docs) & set(relevant_docs))
        
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return precision, recall, f_measure
                                                                          