# Chatbot with Retrieval-Augmented Generation (RAG)

This project builds a chatbot that combines **retrieval-augmented generation (RAG)** and **document grading**. It retrieves relevant documents from a vector store, grades them based on relevance to the user query, and generates responses accordingly. Additionally, the chatbot can perform web searches and re-write queries for better information retrieval. The project also includes a **Streamlit** frontend for a sleek user interface.

## Requirements

- Python 3.8+
- OpenAI API Key
- Tavily API Key
- LangChain
- LangGraph
- Chroma (for vector store)
- Streamlit
- Other necessary Python packages (listed below)

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/chatbot-rag.git
    cd chatbot-rag
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your API keys:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key
    ```

4. Initialize the Chroma vector store in your local environment.

## Usage

1. The core of this chatbot is defined in the `chatbot` class. The class performs multiple functions in a state machine workflow:

   - **Retrieve**: Retrieves relevant documents based on the user query from a Chroma vector store.
   - **Grade Documents**: Grades the documents for relevance to the query and filters out irrelevant ones.
   - **Generate Response**: If relevant documents are found, generates a response using a pre-trained language model (e.g., GPT-3.5).
   - **Transform Query**: Rewrites the query to improve the information retrieval from web searches if necessary.
   - **Net Search**: If documents are not sufficient, performs a web search using Tavily.
   - **Decide to Generate**: Based on document grading, decides whether to generate a response or transform the query for a better search.

2. To run the chatbot via Streamlit, use the following command:

    ```bash
    streamlit run app.py
    ```

3. The chatbot UI will load, and you can enter a question in the input field. After submitting, the chatbot will process the query and display the answer.

## Structure

- **`chatbot.py`**: Main script for the chatbot's logic and state workflow.
- **`app.py`**: Streamlit frontend for the chatbot with custom UI.
- **`requirements.txt`**: Contains all the necessary Python packages.
- **`config.py`**: Configuration file containing necessary parameters for API keys, vector stores, and more.

## Flowchart

1. **Retrieve** relevant documents based on the user query.
2. **Grade** the documents for relevance.
3. **Generate Response** based on the retrieved and relevant documents.
4. **Transform Query** if additional web search is needed.
5. **Net Search**: Perform a web search if required and add new documents to the context.
6. **Decide to Generate**: Determine whether to generate a response or rewrite the query.

## Frontend

The chatbot is deployed via **Streamlit** for an interactive user interface. The UI is styled with custom CSS for readability and modern design:

- **Title**: Displays a bold title with a chat bubble for answers.
- **Text Input**: Allows the user to input questions with a sleek text box.
- **Button**: Users can click to get a response, with a gradient button effect.
- **Chat Bubbles**: Display answers in an elegant chat bubble format.

## Technologies Used

- **LangChain**: Framework for building chains of models and components. 
- **LangGraph**: Framework for workflow management and state graphs.##
- **OpenAI API**: Used for language models (GPT-3.5) to generate responses.
- **Tavily API**: Used to perform web searches when documents are insufficient.
- **Chroma**: Vector database for storing and retrieving document embeddings.
- **Streamlit**: Used for building the user interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork this repository, create issues, or submit pull requests for enhancements or fixes. Contributions are welcome!

## Author

Abhay
