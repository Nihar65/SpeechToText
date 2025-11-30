import argparse, os
from task_extractor.services import InferencePipeline, TaskExtractorService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Audio file or text')
    parser.add_argument('--api-key', help='Deepgram API Key')
    args = parser.parse_args()
    
    pipeline = InferencePipeline(TaskExtractorService(), args.api_key or os.environ.get("DEEPGRAM_API_KEY"))
    
    if args.input.endswith(('.wav', '.mp3')):
        result = pipeline.process_audio(args.input)
    else:
        result = pipeline.process_text(args.input)
    
    print(result)

if _name_ == '_main_':
    main()