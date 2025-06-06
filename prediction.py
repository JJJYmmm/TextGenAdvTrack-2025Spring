import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils.model_utils import adjust_prediction_score

def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
        default="your team name",
        help="Your team name"
    )
    arg.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV dataset",
        default="your dataset path"
    )
    arg.add_argument(
        "--model-type",
        type=str,
        choices=["argugpt", "argugpt-sent", "kerasnlp", "mage", "qwen3", "bino"],
        help="Type of model to use",
        default="your model"
    )
    arg.add_argument(
        "--result-path",
        type=str,
        help="Path to save the results",
        default="your result path"
    )
    arg.add_argument(
        '--part',
        type=int,
        help="Part of the dataset to process 0 or 1",
        default=0
    )
    arg.add_argument(
        '--total',
        type=int,
        help="Total number of parts",
        default=1
    )
    opts = arg.parse_args()
    return opts

def get_dataset(opts):
    print(f"Loading dataset from {opts.data_path}...")
    data = pd.read_csv(opts.data_path)

    # New format: prompt, text
    dataset = data[data['text'].notna()][['prompt', 'text']].copy()
    print(f"Prepared dataset with {len(dataset)} prompts")
    
    return dataset

def get_model(opts):
    print(f"Loading {opts.model_type} detector model...")
    
    if opts.model_type == "argugpt":
        from transformers import pipeline, AutoTokenizer
        model = pipeline(task='text-classification', model='argugpt-roberta', device="cuda", max_length=512, truncation=True)

    elif opts.model_type == 'argugpt-sent':
        from transformers import pipeline, AutoTokenizer
        model = pipeline(task='text-classification', model='argugpt-roberta-sent', device="cuda", max_length=512, truncation=True)

    elif opts.model_type == 'kerasnlp':
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        import keras_nlp
        import keras_core as keras 
        import keras_core.backend as K
        vocab_path = 'vocab.spm'
        ckpt_path = 'fold0.keras'
        tokenizer = keras_nlp.models.DebertaV3Tokenizer(vocab_path)
        tokenizer = keras_nlp.models.DebertaV3Preprocessor(tokenizer, sequence_length=200)
        import keras_nlp.src.models.deberta_v3.deberta_v3_classifier
        model = keras.models.load_model(
            ckpt_path,
            compile=False
        )
        model = (model, tokenizer)

    elif opts.model_type == 'mage':
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('yaful/MAGE')
        model = AutoModelForSequenceClassification.from_pretrained('yaful/MAGE', device_map = "cuda")
        model = (model, tokenizer)
    
    elif opts.model_type == 'qwen3':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        model = (model, tokenizer)
    
    elif opts.model_type == 'bino':
        from binoculars import Binoculars
        model = Binoculars()

    print("Model loaded successfully")
    return model

def predict_keras(text, model, tokenizer):
    text = tokenizer(text)  # Preprocess text
    text['token_ids'] = text['token_ids'].unsqueeze(0)
    text['padding_mask'] = text['padding_mask'].unsqueeze(0)
    prediction = model.predict(text)
    score = prediction[0][0].item()
    return 1 - score

def predict_argugpt(text, model):
    prediction = model(text)
    pred_label = prediction[0]['label']
    pred_score = prediction[0]['score']
    # # Adjust score - higher values indicate more likely human-written text
    final_score = adjust_prediction_score(pred_label, pred_score, 'argugpt')
    best_alpha = 1.60
    best_beta = -0.87
    final_score = best_alpha * final_score + best_beta
    return final_score

import torch
from deployment import preprocess, detect
def predict_mage(text, model, tokenizer):
    text = preprocess(text)
    tokenize_input = tokenizer(text, truncation=True)
    tensor_input = torch.tensor([tokenize_input["input_ids"]]).to(model.device)
    outputs = model(tensor_input)
    th=-3.08583984375
    is_machine = -outputs.logits[0][0].item()
    score = is_machine - th + 0.5
    return score

def predict_qwen3(prompt, text, model, tokenizer):
    '''
      use batch infer or vllm is much much faster
    '''
    prompt = "You are an expert in detecting AI-generated text. Given a pair of instruction and response, you need to determine whether the response was written by a human or generated by an AI. Return a single number between 0 and 1, rounded to two decimal places, indicating the probability that the response was written by a human."
    text = f"Instruction: {prompt}\nResponse: {text}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # Extract the probability from the content
    prob = 1
    try:
        prob = float(content.strip())
    except ValueError:
        pass

    print(thinking_content, content, prob)

    return prob

def predict_qwen3_batch(prompts, texts, model, tokenizer):
    probs = []
    all_messages = []

    for prompt, text in zip(prompts, texts):
        full_prompt = "You are an expert in detecting AI-generated text. Given a pair of instruction and response, you need to determine whether the response was written by a human or generated by an AI. Return a single number between 0 and 1, rounded to two decimal places, indicating the probability that the response was written by a human."
        input_text = f"Instruction: {prompt}\nResponse: {text}"
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": input_text}
        ]
        all_messages.append(messages)

    batched_inputs = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for messages in all_messages
    ]

    model_inputs = tokenizer(batched_inputs, return_tensors="pt", padding=True, truncation=False).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )

    for i in range(len(all_messages)):
        input_len = len(model_inputs.input_ids[i])
        output_ids = generated_ids[i][input_len:].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        prob = 1.0
        try:
            prob = float(content.strip())
        except ValueError:
            pass

        probs.append(prob)

    return probs

def predict_bino_batch(texts, model):
    return model.compute_score(texts)

def run_prediction_batch(model, dataset, model_type, opts_part=0, opts_total=1, batch_size=4):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    texts = dataset['text'].tolist()

    total_length = len(prompts)
    if opts_total > 1:
        start_idx = int(opts_part * total_length / opts_total)
        end_idx = int((opts_part + 1) * total_length / opts_total)
        prompts = prompts[start_idx:end_idx]
        texts = texts[start_idx:end_idx]
    
    text_predictions = []
    
    start_time = pd.Timestamp.now()
    
    print("Processing texts in batches...")
    num_samples = len(prompts)
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Batch predictions"):
        batch_prompts = prompts[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        # 批量预测
        if model_type == 'qwen3':
            batch_probs = predict_qwen3_batch(
                prompts=batch_prompts,
                texts=batch_texts,
                model=model[0],
                tokenizer=model[1]
            )
        elif model_type == 'bino':
            batch_probs = predict_bino_batch(
                texts=batch_texts,
                model=model
            )
        else:
            raise ValueError(f"Invalid model type for batch infer: {model_type}")
        
        text_predictions.extend(batch_probs)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    results_data = {
        'prompt': prompts,
        'text_prediction': text_predictions
    }
    
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

def run_prediction(model, dataset, model_type, opts_part = 0, opts_total = 1):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    texts = dataset['text'].tolist()

    total_length = len(prompts)
    if opts_total > 1:
        start_idx = int(opts_part * total_length / opts_total)
        end_idx = int((opts_part + 1) * total_length / opts_total)
        prompts = prompts[start_idx:end_idx]
        texts = texts[start_idx:end_idx]
    
    text_predictions = []
    
    start_time = pd.Timestamp.now()
    
    # Process texts
    print("Processing texts...")
    for idx, text in enumerate(tqdm(texts, desc="Text predictions")):

        predict_func = {
            'argugpt': predict_argugpt,
            'argugpt-sent': predict_argugpt,
            'kerasnlp': predict_keras,
            'mage': predict_mage,
            'qwen3': predict_qwen3
        }.get(model_type)

        if model_type != 'qwen3':
            final_score = predict_func(text, model) if type(model) != tuple else predict_func(text, model[0], model[1])
        else:
            final_score = predict_qwen3(prompts[idx], text, model[0], model[1])
        text_predictions.append(final_score)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Create results in the requested format
    results_data = {
        'prompt': prompts,
        'text_prediction': text_predictions
    }
    
    # Create results dictionary
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    model = get_model(opts)

    if opts.model_type != 'qwen3':
        results = run_prediction(model, dataset, opts.model_type, opts.part, opts.total)
    else: # for qwen3, we use batch infer
        results = run_prediction_batch(model, dataset, opts.model_type, opts.part, opts.total)
    
    # Save results
    os.makedirs(opts.result_path, exist_ok=True)

    if opts.total > 1:
        writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + f'_{opts.part}_{opts.total}' + ".xlsx"), engine='openpyxl')
    else:
        writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')

    # Create prediction dataframe with the required columns
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    
    # Filter out rows with None values
    prediction_frame = prediction_frame[prediction_frame['text_prediction'].notnull()]
    print(len(prediction_frame))
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '.xlsx')}")
