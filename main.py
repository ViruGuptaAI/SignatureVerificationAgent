# Before running the sample:
#    pip install --pre azure-ai-projects>=2.0.0b1
#    pip install azure-identity

import base64
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai.types.responses import ResponseInputTextParam, ResponseInputImageParam
import os
from dotenv import load_dotenv
from websockets import Data

load_dotenv(override=True)

myEndpoint = os.getenv("agentEndpoint")

project_client = AIProjectClient(
    endpoint=myEndpoint,
    credential=DefaultAzureCredential(),
)

myAgent = "SignatureMatcher"
# Get an existing agent
agent = project_client.agents.get(agent_name=myAgent)
print(f"Retrieved agent: {agent.name}")

openai_client = project_client.get_openai_client()


def SignatureMatcher(image1, image2):

    with open(image1, "rb") as f:
        image_data1 = base64.standard_b64encode(f.read()).decode("utf-8")

    with open(image2, "rb") as f:
        image_data2 = base64.standard_b64encode(f.read()).decode("utf-8")


    # Reference the agent to get a response
    response = openai_client.responses.create(
        input=[
            {
                "role": "user",
                "content": [
                    ResponseInputTextParam(type="input_text", text="Analyze both the signatures in the images and determine if they match."),
                    ResponseInputImageParam(type="input_image", image_url=f"data:image/png;base64,{image_data1}"),
                    ResponseInputImageParam(type="input_image", image_url=f"data:image/png;base64,{image_data2}"),
                ],
            }
        ],
        extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
    )

    # print(f"Response output: {response.output_text}")

    return response.output_text



if __name__ == "__main__":
    image1 = "./Data/VR1.jpg"
    image2 = "./Data/VR2.jpg"

    output = SignatureMatcher(image1, image2)
    print(f"Signature matching result: {output}") 