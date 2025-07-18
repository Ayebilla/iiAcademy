import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from tools.salary_tool import salary_prediction_tool
from langchain.agents.agent_types import AgentType 
import os
from dotenv import load_dotenv
import json 


# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a career advisor that helps with:
#     1. Career path suggestions
#     2. Salary predictions (use the salary_prediction_tool)
#     3. Job market insights"""),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are CareerPath Pro, an insightful and empathetic career advisor. Your role is to:

1. Provide thoughtful career guidance with a global perspective
2. Offer detailed salary comparisons using the SalaryPredictor tool
3. Help users understand career growth opportunities

When discussing salaries:
- Always use the SalaryPredictor tool for accurate predictions
- Compare salaries across countries when relevant
- Explain how factors like experience, education, and location affect earnings
- Suggest career moves that could improve compensation

For salary predictions:
- First ask for all required parameters (age, gender, education, job title, experience, country, race)
- Make predictions for different scenarios when helpful (e.g., "If you moved to the US..." or "With 5 more years experience...")
- Put numbers in context (e.g., cost of living differences)

Tone:
- Professional yet approachable
- Culturally aware
- Data-driven but not robotic
- Encouraging about career growth

Example responses:
"Based on your profile in Ghana, your estimated salary is $X. For comparison, someone with similar qualifications in the US typically earns $Y. Would you like to explore what skills could help bridge this gap?"

"Your current predicted salary is $A. If you complete that master's degree, we could see that increase to approximately $B. Would you like me to explain how?"

"Interesting! As a {job_title} in {country}, you're earning ${amount}. In {comparison_country}, the range is typically ${range}. This difference often reflects {reason}." 
"""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

load_dotenv()

@cl.on_chat_start
async def start_chat():
    try:
        llm = ChatOpenAI(openai_api_key=" ")
        
        tools = [
            Tool(
                name="SalaryPredictor",
                func=salary_prediction_tool,
                description="Predicts salary. Requires: age, gender, education, job_title, experience, country"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,  # Correct enum usage
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        cl.user_session.set("agent", agent)
        await cl.Message(content="Career advisor ready! Try: 'Predict salary for a 30-year-old male Software Engineer with 5 years experience in the US'").send()
    
    except Exception as e:
        await cl.Message(content=f"Startup error: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        
        print("Received message content:", message.content)
        print("Full message object:", message)

        if any(keyword in message.content.lower() for keyword in ["salary", "predict", "earn"]):
            try:
                input_msg = await cl.AskUserMessage(
                    content="Please provide your details in the following format (one per line):\n\n"
                            "Age: [value]\nGender: [value]\nEducation: [value]\n"
                            "Job Title: [value]\nExperience: [value]\nCountry: [value]\nrace: [value]",
                    timeout=300
                ).send()

                if input_msg:
                    try:
                        print("User responded with:", input_msg)  # Log user's form reply
                        
                        content = input_msg["output"].strip()

                        input_lines = content.split('\n')

                        if len(input_lines) < 6:
                            raise ValueError("Incomplete input: Need all 6 fields")

                        params = {
                            "age": input_lines[0].split(':')[1].strip(),
                            "gender": input_lines[1].split(':')[1].strip(),
                            "education": input_lines[2].split(':')[1].strip(),
                            "job_title": input_lines[3].split(':')[1].strip(),
                            "experience": input_lines[4].split(':')[1].strip(),
                            "country": input_lines[5].split(':')[1].strip(),
                            "race": input_lines[6].split(':')[1].strip()
                        }

                        print("Parsed parameters:", params)

                        result = salary_prediction_tool(json.dumps(params))
                        await cl.Message(content=result).send()

                    except IndexError:
                        await cl.Message(content="Please make sure each line has a colon followed by a value.").send()
                    except ValueError as ve:
                        await cl.Message(content=str(ve)).send()
                    except Exception as e:
                        await cl.Message(content=f"Error processing your input: {str(e)}").send()
                    return

            except Exception as e:
                await cl.Message(content=f"Failed to collect input: {str(e)}").send()
                return

        # Default agent fallback for other questions
        try:
            agent = cl.user_session.get("agent")
            if not agent:
                raise Exception("Agent not initialized")

            response = await agent.arun(input=message.content)
            await cl.Message(content=response).send()

        except Exception as e:
            await cl.Message(content=f"Error processing your request: {str(e)}").send()

    except Exception as e:
        await cl.Message(content=f"Unexpected error: {str(e)}").send()
