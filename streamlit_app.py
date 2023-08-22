import streamlit as st
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json

st.set_page_config(layout='wide', page_title="ScrapeIt AI", page_icon=":computer:")
col1, col2 = st.columns(2)
with col1:
    with open("logo.svg", "r") as file:
        svg_logo = file.read()
    st.markdown(svg_logo, unsafe_allow_html=True)
st.subheader("AI-powered Web Scraping")
st.divider()




url = st.sidebar.text_input('Enter the URL you want to scrape:')
api_key = st.sidebar.text_input('Enter your OpenAI API Key', type="password")

st.sidebar.divider()
element_one = st.sidebar.text_input('Enter which element you want to scrape')
option_one = st.sidebar.selectbox(
    'Which data type is the first element',
    ('string', 'integer'))
st.sidebar.divider()
element_two = st.sidebar.text_input('Enter another element you want to scrape')
option_two = st.sidebar.selectbox(
    'Which data type is the second element',
    ('string', 'integer'))



if not api_key:
    st.warning("Please enter your OpenAI API key!")
    st.stop()

st.divider()


async def async_load_playwright(url) -> str:
    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
         
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results
# langchain feature
async def main():
    output = await async_load_playwright(url)

    # langchain part
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
    from langchain.prompts import ChatPromptTemplate

    with st.spinner('Running extraction and loading data...'):
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=api_key)
        schema = {
            "properties": {
                element_one: {"type": option_one},
                element_two: {"type": option_two},
            },
            "required": [element_one, element_two],
        }
        chain = create_extraction_chain(schema, llm)
        results = chain.run(output[:3500])
        df = pd.DataFrame(results)
    
    
    st.table(df)
    df = pd.DataFrame(results)

    json_results = json.dumps(results).encode('utf-8')
    st.sidebar.divider()
    st.sidebar.download_button(
        label="Download JSON",
        data=json_results,
        file_name="scrapeit_data.json",
        mime="application/json"  
    )

    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download as CSV",
        data=csv,
        file_name="scrapeit_data.csv",
        mime="text/csv" 
    )

if __name__ == '__main__':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    title=loop.run_until_complete(main())


