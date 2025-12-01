import argparse, os
# Note: Ensure the folder is named 'service' (singular) as per Step 1
from task_extractor.services import InferencePipeline, TaskExtractorService

def main():
    parser = argparse.ArgumentParser(description="Audio to Task Extraction")
    parser.add_argument('input', help='Audio file path or text content')
    parser.add_argument('--api-key', help='Deepgram API Key')
    args = parser.parse_args()
    
    # Initialize the pipeline with the service
    pipeline = InferencePipeline(
        TaskExtractorService(), 
        args.api_key or os.environ.get("DEEPGRAM_API_KEY")
    )
    
    # Process based on input type (file extension check)
    if args.input.endswith(('.wav', '.mp3', '.m4a', '.mp4')):
        result = pipeline.process_audio_file(args.input)
    else:
        result = pipeline.process_text(args.input)
    
    # Print the result in a readable format
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))

if __name__ == '__main__':
    main()