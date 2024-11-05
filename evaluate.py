import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import semantic_similarity as ss

load_dotenv()


def evaluate_with_similarity(eval_data, similarity_fn, threshold=0.8):
    similarity_results = similarity_fn(
        df=eval_data, 
        column_1='Producer Name_x', 
        column_2='Producer Name_y', 
        sample_size=eval_data.shape[0], 
        new_col_name='similarity'
    )

    similarity_results['pred'] = similarity_results['similarity'] > threshold
    return similarity_results


class NameTest(BaseModel):
    """
    Assess whether two names of cocoa producers in Cote D'Ivoire are the same.
    """
    is_same_entity: bool = Field(description="True if the names describe the same entity, false otherwise.")
        
class CoopNameHomogenizer:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(temperature=temperature, model=model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert on cocoa supply chains in Cote D'Ivoire. "
                    "You are tasked with assessing whether the names of various Cocoa cooperatives are representing the same entity. "
                    "Names can be the same but still represent different entities. Pay special attention to the abbreviations. "
                    "Abbreviation representing the same entity will have similar content disregarding punctuation and whitespace. "
                ),
                (
                    "human",
                    "Are these two names representing the same entity? "
                    "Respond with True or False.\n\nName 1: {name1}\nName 2: {name2}"
                    "\nAbbreviation 1: {abbrev1}\nAbbreviation 2: {abbrev2}"
                ),
            ]
        )
        self.classifier = self.prompt | self.llm.with_structured_output(
            schema=NameTest,
            method="function_calling",
            include_raw=False,
        )

    def test_pair(self, pair_data):
        return self.classifier.invoke({
            "name1": pair_data[0], 
            "name2": pair_data[1], 
            "abbrev1": pair_data[2], 
            "abbrev2": pair_data[3]
        })

    def run_all(self, data, sample_size=None):
        if sample_size is not None:
            data = data.sample(n=sample_size, random_state=1)
        data["pred"] = data.apply(lambda row: self.test_pair(row).is_same_entity, axis=1)
        return data

    def evaluate(self, input_csv_path, output_csv_path):
        eval_data = pd.read_csv(input_csv_path)
        results = self.run_all(eval_data)
        results.to_csv(output_csv_path)
        return results

    
if __name__ == "__main__":
#     # Evaluate gpt-4o-mini with no examples
#     evaluator = CoopNameHomogenizer(model_name="gpt-4o-mini")
#     evaluator.evaluate(
#         'data/eval/hand_annotated_pairs.csv', 
#         'data/outputs/gpt_4o_mini_no_examples_eval.csv')
    
#     # Evaluate gpt-3.5-turbo with no examples
#     evaluator = CoopNameHomogenizer(model_name="gpt-3.5-turbo")
#     evaluator.evaluate(
#         'data/eval/hand_annotated_pairs.csv', 
#         'data/outputs/gpt_35_turbo_no_examples_eval.csv')


    # Evaluate Similarity Models 
    eval_data = pd.read_csv('data/eval/hand_annotated_pairs.csv')
    
    # Evalueate semantic similarity 
    semantic_similarity_results = evaluate_with_similarity(
        eval_data=eval_data, 
        similarity_fn=ss.process_semantic_similarity,
        threshold=0.88)
    semantic_similarity_results.to_csv('data/outputs/semantic_similarity_eval.csv')
    
    # Evaluate semantic similarity using only second half of each name
    second_half_similarity_results = evaluate_with_similarity(
        eval_data=eval_data, 
        similarity_fn=ss.process_second_half_similarity,
        threshold=0.88)
    second_half_similarity_results.to_csv('data/outputs/second_half_similarity_eval.csv')
    
    # Evaluate TF-IDF vector similarity 
    tf_idf_similarity_results = evaluate_with_similarity(
        eval_data=eval_data, 
        similarity_fn=ss.process_tf_idf,
        threshold=0.88)
    tf_idf_similarity_results.to_csv('data/outputs/tf_idf_similarity_eval.csv')
    
    