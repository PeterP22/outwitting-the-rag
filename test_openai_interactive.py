#!/usr/bin/env python3
"""Interactive OpenAI chat with better timeout handling"""

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
import signal

# Load environment variables
load_dotenv()

def timeout_handler(signum, frame):
    print("\n⏰ Request timed out. The model might be taking too long to respond.")
    print("Try a simpler question or press Ctrl+C to exit.")

def ask_question(client, model, question):
    """Ask a single question with timeout handling"""
    
    print("\n🔄 Processing... ", end='', flush=True)
    start_time = time.time()
    
    try:
        # Set a 30-second alarm
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_completion_tokens=1000,
            timeout=25  # API timeout slightly less than signal timeout
        )
        
        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        elapsed = time.time() - start_time
        print(f"Done! ({elapsed:.1f}s)")
        
        # Extract answer
        answer = response.choices[0].message.content
        
        if answer and answer.strip():
            print(f"\n💬 Answer: {answer}")
        else:
            print(f"\n💬 Answer: [Empty response]")
            # Show token details if empty
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                print(f"   (Used {details.reasoning_tokens} reasoning tokens)")
        
        # Show token usage
        if hasattr(response, 'usage') and response.usage:
            print(f"\n🔢 Tokens: {response.usage.total_tokens} total "
                  f"({response.usage.prompt_tokens} prompt, "
                  f"{response.usage.completion_tokens} completion)")
        
    except Exception as e:
        # Cancel any pending alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        print(f"\r❌ Error: {str(e)}")

def main():
    """Run interactive chat"""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4")
    
    print(f"\nOpenAI Chat - Interactive Mode")
    print(f"Model: {model}")
    print("=" * 60)
    print("Tips:")
    print("- Type your questions and press Enter")
    print("- Type 'quit' or 'exit' to stop")
    print("- Press Ctrl+C anytime to exit")
    print("- If a response takes too long, it will timeout after 30s")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Process the question
            ask_question(client, model, question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()