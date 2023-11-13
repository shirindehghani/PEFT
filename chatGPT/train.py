import openai
import json

def get_api_keys(key:str):
    openai.api_key=key
    return None

def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

def fine_tune(dataset_train, dataset_validation, model_name="gpt-3.5-turbo",needed_val=True):
    if needed_val:
        write_jsonl(dataset_train, "train.jsonl")
        write_jsonl(dataset_validation, "validation.jsonl")
    else:
        write_jsonl(dataset_train, "train.jsonl")
        
    training_response = openai.File.create(file=open("./train.jsonl", "rb"), purpose="fine-tune")
    training_file_id = training_response["id"]

    validation_response = openai.File.create(file=open("./validation.jsonl", "rb"), purpose="fine-tune")
    validation_file_id = validation_response["id"]

    response = openai.FineTuningJob.create(
        training_file=training_file_id, validation_file=validation_file_id,
        model=model_name,suffix="chat-bot")
    job_id = response["id"]
    response = openai.FineTuningJob.retrieve(job_id)
    return response


