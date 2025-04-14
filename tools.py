import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


def inference_batch(texts, model, tokenizer, device, batch_size=100):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, padding_side='left').to(device)
            input_length = inputs['input_ids'].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                no_repeat_ngram_size=3,  # Evita repetição de 2-gramas
                do_sample=True,  # Ativa amostragem controlada
                top_k=50,  # Limita o vocabulário considerado
                top_p=0.95,  # Nucleus sampling
                temperature=0.9,  # Controla a aleatoriedade
                # repetition_penalty=1.5,  # Penaliza repetições
                eos_token_id=tokenizer.eos_token_id  # Ponto de parada natural
            )
            generated_ids = outputs.sequences[:, input_length:]
            batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_predictions = [prediction.strip() for prediction in batch_predictions]
            predictions.extend(batch_predictions)
            progress_bar = '#' * (i//batch_size) + ' ' * ((len(texts)//batch_size) - (i//batch_size) - 1)
            print(f"\rProgress: [{progress_bar}] [{i + batch_size}/{len(texts)}]", end='')
    return predictions


def get_model(device):
    model_name = 'ComCom/gpt2-small'
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def tokenize_function(example, prompt, tokenizer):
    # Construir o input_text (sem incluir o output)
    input_text = prompt.format(example['instruction'], example['input'])
    
    # Tokenizar o input_text
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Tokenizar o output para criar os labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example['output'],
            truncation=True,
            max_length=512,
            padding="max_length"
        ).input_ids
    
    # Substituir tokens de padding nos labels por -100
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    
    # Adicionar os labels ao dicionário model_inputs
    model_inputs["labels"] = labels
    
    return model_inputs


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def format_inference_shots(inference_dataset, n_shots):
    if n_shots > 0:
        selected_shots = inference_dataset.select(range(0, n_shots))

        explatation = f"Below are {n_shots} instructions that describe task resolutions. First comes ### Instruction, providing the instruction to be followed. Second comes ### Input, providing information that supports the instruction. And last comes ### Response, bringing the response to this instruction."
        prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}

"""

        inference_shots = ''.join([prompt.format(i['instruction'], i['input'], i['output']) for i in selected_shots])
        inference_shots = explatation + inference_shots
        return inference_shots
    else:
        return ''


def format_input_texts(instructions_dataset, inference_shots=''):
    end_prompt = """Below is an instruction for a task. First comes ### Instruction, giving the instruction to be followed. Second comes ### Input, giving information that supports the instruction. Write a response that adequately completes the request from ### Response.
### Instruction:
{}

### Input:
{}

### Response:
"""

    input_texts = [inference_shots + '\n' +
                   end_prompt.format(data['instruction'], data['input'])
                   for data in instructions_dataset]
    
    return input_texts
