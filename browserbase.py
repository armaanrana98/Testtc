import os
from crewai.tools import tool
from playwright.sync_api import sync_playwright
from html2text import html2text
from time import sleep

@tool("Browserbase tool")
def browserbase(url: str):
    """
    Loads a URL using a headless web browser via Browserbase.
    
    Uses hardcoded API credentials.
    
    :param url: The URL to load.
    :return: The text content of the rendered page.
    """
    # Hardcoded credentials
    api_key = "bb_live_6hNs3hAALOJxLOrYgAC7HR9LXxE"
    project_id = "aa782a6e-a3fd-4208-b472-28ecf82711e1"
    
    connection_url = (
        "wss://connect.browserbase.com?"
        f"apiKey={api_key}&projectId={project_id}"
    )
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(connection_url)
        # Get the first available context and its first page.
        context = browser.contexts[0]
        page = context.pages[0]
        page.goto(url)
        # Wait for dynamic content to load (adjust sleep duration if necessary)
        sleep(25)
        content = html2text(page.content())
        browser.close()
        return content
