'''
Contains logic for evaluating models; outputs results to CSV files when called from command line 
'''

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import semantic_similarity as ss
from tqdm import tqdm
import tiktoken

load_dotenv()


def evaluate_with_similarity(eval_data, similarity_fn, output_path='data/test-results/similarity_eval.csv'):
    """
    Evaluates results from similarity estimate using a threshold to binarize model output. 
    """
    similarity_results = similarity_fn(
        df=eval_data, 
        column_1='Producer Name_x', 
        column_2='Producer Name_y', 
        sample_size=eval_data.shape[0], 
        new_col_name='similarity'
    )

    threshold = find_best_threshold(similarity_results)[0]

    similarity_results['pred'] = similarity_results['similarity'] > threshold
    similarity_results.to_csv(output_path, index=False)
    return similarity_results


class PairAssessment(BaseModel):
    """
    Assess whether two names of cocoa producers in Cote D'Ivoire are the same.
    """
    is_same_entity: bool = Field(description="True if the names describe the same entity, false otherwise. This is not optional.")
     
class CoopNameHomogenizer:
    """
    Class for using LLMs to assess whether pairs of Cocoa Cooperative datapoints are the same or different. 
    """
    def __init__(self, model_name="gpt-4o-mini", temperature=0, path_to_examples=""):
        self.model_name = model_name
        self.llm = ChatOpenAI(temperature=temperature, model=model_name)
        # Set examples for LLM if passed in
        if path_to_examples == "":
            self.examples = ""
        else:
            with open(path_to_examples, 'r') as file:
                self.examples = file.read()
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert on cocoa supply chains in Cote D'Ivoire. "
                    "You are tasked with assessing whether the name and abbreviation of a pair of cocoa cooperatives are representing the same entity. "
                    "Names can be the same but still represent different entities. Pay special attention to the abbreviations. ",
                ),
                (
                    self.examples
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
            schema=PairAssessment,
            method="function_calling",
            include_raw=False,
        )

        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_input_tokens(self, pair_data):
        """
        Returns the number of tokens in the input sent to the classifier for a given query.
        """
        if not type(pair_data) == list:
            pair_data = pair_data.to_list()
        # Build the prompt with the query data to estimate token count
        query_input = {
            "name1": pair_data[0], 
            "name2": pair_data[1], 
            "abbrev1": pair_data[2], 
            "abbrev2": pair_data[3]
        }
        formatted_prompt = self.prompt.format_messages(**query_input)
        
        # Concatenate all parts of the formatted prompt to get the full input string
        full_input_text = ''.join([message.content for message in formatted_prompt])

        # Use the LLM tokenizer to count tokens in the full input text
        token_count = len(self.tokenizer.encode(full_input_text))
        return token_count


    def test_pair(self, pair_data):
        if not type(pair_data) == list:
            pair_data = pair_data.to_list()
        # print(pair_data)
        try:
            return self.classifier.invoke({
                "name1": pair_data[0], 
                "name2": pair_data[1], 
                "abbrev1": pair_data[2], 
                "abbrev2": pair_data[3]
            })
        except ValidationError as exc:
            print(repr(exc.errors()[0]['type']))


    def run_all(self, data, sample_size=None):
        if sample_size is not None:
            data = data.sample(n=sample_size, random_state=1)
        
        # Add a progress bar using tqdm
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating {self.model_name}"):
            result = self.test_pair(row).is_same_entity
            results.append(result)
        data["pred"] = results
        return data

    def evaluate(self, input_csv_path, output_csv_path):
        eval_data = pd.read_csv(input_csv_path)
        results = self.run_all(eval_data)
        results.to_csv(output_csv_path, index=False)
        return results



def evaluate_llm_model(model_name, input_csv, output_csv, examples_path=None):
    evaluator = CoopNameHomogenizer(model_name=model_name, path_to_examples=examples_path)
    evaluator.evaluate(input_csv_path=input_csv, output_csv_path=output_csv)


def find_best_threshold(df):
    """
    Find the best threshold for similarity scores to classify pairs of records as representing the same entity.

    Parameters:
    csv_file (str): Path to the CSV file containing the data. The file must include the following columns:
        - 'similarity': The similarity scores between record pairs.
        - 'classification': The ground truth labels indicating if the records represent the same entity (1) or not (0).

    Returns:
    tuple: A tuple containing:
        - best_threshold (float): The similarity score threshold that yields the highest F1-score.
        - best_f1 (float): The highest F1-score achieved.
    """

    # Ensure the necessary columns exist
    if not {'similarity', 'classification'}.issubset(df.columns):
        raise ValueError("The CSV file must contain 'similarity' and 'classification' columns.")

    # Sort the similarity scores to define thresholds
    thresholds = np.sort(df['similarity'].unique())

    best_threshold = None
    best_f1 = 0

    # Iterate through thresholds to calculate performance metrics
    for threshold in thresholds:
        # Predict based on the current threshold
        df['pred'] = df['similarity'] >= threshold

        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['classification'], df['pred'], average='binary'
        )

        # Update the best threshold if the F1-score improves
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Return the best threshold and F1-score
    return float(best_threshold), float(best_f1)


if __name__ == "__main__":
    eval_data_path = 'data/eval/hand_annotated_pairs.csv'

    ### Evaluate LLM Based Models ###
    llm_models = [
        #{"model_name": "gpt-4o-mini", "examples_path": "", "output_csv": "data/test-results/gpt_4o_mini_no_examples_eval.csv"},
        #{"model_name": "gpt-4o-mini", "examples_path": "prompts/ll-examples.txt", "output_csv": "data/test-results/gpt_4o_mini_with_examples_eval.csv"},
        # {"model_name": "gpt-4o-mini", "examples_path": "prompts/pseudo-examples.txt", "output_csv": "data/test-results/gpt_4o_mini_with_pseudo_examples_eval.csv"},   
        # {"model_name": "gpt-4o-mini", "examples_path": "prompts/overfitting-gpt4o-original-included.txt", "output_csv": "data/test-results/gpt_4o_mini_overfit_with_original.csv"},
        # {"model_name": "gpt-4o-mini", "examples_path": "prompts/overfitting-gpt4o-pure.txt", "output_csv": "data/test-results/gpt_4o_mini_overfit_pure_eval.csv"},
        # {"model_name": "gpt-4o-mini", "examples_path": "prompts/examples-generated-from-overfit-set.txt", "output_csv": "data/test-results/gpt_4o_mini_generated_examples_eval.csv"},
     

        #{"model_name": "gpt-4o", "examples_path": "", "output_csv": "data/test-results/gpt_4o_no_examples_eval.csv"},
        # {"model_name": "gpt-4o", "examples_path": "prompts/ll-examples.txt", "output_csv": "data/test-results/gpt_4o_with_examples_eval.csv"},
        # {"model_name": "gpt-4o", "examples_path": "prompts/pseudo-examples.txt", "output_csv": "data/test-results/gpt_4o_with_pseudo_examples_eval.csv"},
        # {"model_name": "gpt-4o", "examples_path": "prompts/overfitting-gpt4o-pure.txt", "output_csv": "data/test-results/gpt_4o_overfit_pure_eval.csv"},
        # {"model_name": "gpt-4o", "examples_path": "prompts/overfitting-gpt4o-original-included.txt", "output_csv": "data/test-results/gpt_4o_overfit_with_original.csv"},
        # {"model_name": "gpt-4o", "examples_path": "prompts/examples-generated-from-overfit-set.txt", "output_csv": "data/test-results/gpt_4o_generated_examples_eval.csv"},

        #{"model_name": "gpt-3.5-turbo", "examples_path": "", "output_csv": "data/test-results/gpt_35_turbo_no_examples_eval.csv"},
        #{"model_name": "gpt-3.5-turbo", "examples_path": "prompts/ll-examples.txt", "output_csv": "data/test-results/gpt_35_turbo_with_examples_eval.csv"},
        #{"model_name": "gpt-3.5-turbo", "examples_path": "prompts/pseudo-examples.txt", "output_csv": "data/test-results/gpt_35_turbo_with_pseudo_examples_eval.csv"},
        # {"model_name": "gpt-3.5-turbo", "examples_path": "prompts/examples-generated-from-overfit-set.txt", "output_csv": "data/test-results/gpt_35_turbo_generated_examples_eval.csv"},
    ]

    for model in tqdm(llm_models, desc="Evaluating LLM Models"):
        evaluate_llm_model(
            model_name=model["model_name"],
            input_csv=eval_data_path,
            output_csv=model["output_csv"],
            examples_path=model["examples_path"]
        )

    ### Evaluate Similarity Models ###
    eval_data = pd.read_csv(eval_data_path)
    similarity_functions = [
        {"similarity_fn": ss.process_semantic_similarity, "output_csv": "data/test-results/semantic_similarity_eval.csv"},
        {"similarity_fn": ss.process_second_half_similarity, "output_csv": "data/test-results/second_half_similarity_eval.csv"},
        {"similarity_fn": ss.process_tf_idf, "output_csv": "data/test-results/tf_idf_similarity_eval.csv"}
    ]

    for sim in tqdm(similarity_functions, desc="Evaluating Similarity Models"):
        evaluate_with_similarity(
            eval_data=eval_data,
            similarity_fn=sim["similarity_fn"],
            output_path=sim["output_csv"]
        )
    
    