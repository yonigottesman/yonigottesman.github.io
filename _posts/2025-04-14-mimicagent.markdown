---
layout: post
title:  A Minimal Agent to Explore Mimic-III Dataset
excerpt: Building a Framework-Free SQL Agent for Exploring MIMIC-III Medical Data with Vector-Based Query Retrieval.
date:   2025-04-14 00:00:00 +0000
categories: []
hide: false
---

Looking at your agent run for the first time is a truly magical experience. Of course I get shocked by any good reasoning response by an LLM, but for some reason, watching an agent choose what tool to run, give it the output, then watching it choose the next tool, feels different. It feels a bit like in the old days of 2022 ChatGPT when we were in shock an LLM can write a haiku.  

Implementing agents is also kind of a unique experience, instead of me figuring out the flow of my program, I only need to add more tools and let the agent decide what to use when. But how should we implement agents in 2025? Every time I login into Twitter I hear about another agent library. You have [agno](https://github.com/agno-agi/agno/tree/main), [langgraph](https://python.langchain.com/docs/concepts/architecture/#langgraph), [smolagents](https://github.com/huggingface/smolagents), [crewAI](https://github.com/crewAIInc/crewAI), [llamaindex](https://docs.llamaindex.ai/en/stable/) and even [pydantic](https://ai.pydantic.dev/agents/) (wait what?!).
All of these are probably legit, but using another dependency comes with a price, too many things happen behind the scene without me understanding what's going on. I'm not saying I understand everything in the stack between my python code and the assembly running on my CPU, but I think this new "agentic" way of work is too immature for us to already start hiding lots of abstractions from us.

In this post, I'll implement my own (minimal!) agent abstraction, and use it to create an agent that can explore the [mimic-III clinical database](https://physionet.org/content/mimiciii/1.4/). Find my implementation [here](https://github.com/yonigottesman/mimic-agent).

### A Minimal Abstraction
An agent needs some tools to interact with the world. Every tool is just a function with some input schema. Every library chooses a different way to abstract a "Tool" - in my case, a tool is a function with the following:
* A docstring - This will be passed to the LLM as the tool_description
* An `inputs` parameter with a `pydantic.BaseModel` type - This will be passed to the LLM as the `input_schema`.
Here is an example function:

```python
class SearchWebCommand(BaseModel):
    query: str = Field(..., description="The query to search the web for")
    search_type: Literal["text", "news"] = Field(..., description="The type of search")
    max_results: int = Field(..., description="The maximum number of results")


def search_web(inputs: SearchWebCommand):
    """Search the web using DuckDuckGo."""
    if inputs.search_type == "text":
        return str(DDGS().text(inputs.query, max_results=7))
    elif inputs.search_type == "news":
        return str(DDGS().news(inputs.query, max_results=7))
    else:
        raise ValueError(f"Invalid search type: {inputs.search_type}")
```
To use this function as a tool Ill wrap it in the `Tool` class:

```python
Tool(search_web)
```

The Tool class will know how to extract the docstring and the `inputs`.

```python
class Tool:
    def __init__(self, function: Callable, call_args=None):
        if function.__doc__ is None:
            raise ValueError("Tool functions must have a docstring")

        self.function = function
        self.description = function.__doc__
        self.name = function.__name__
        if "inputs" not in get_type_hints(function):
            self.input_model = None
            self.input_schema = {"type": "object", "properties": {}, "required": []}
        else:
            self.input_model = get_type_hints(function)["inputs"]
            self.input_schema = self.input_model.model_json_schema()

        self.additional_args = call_args or {}
```

I really like how smolagents use a decorator `@tool` to make a function into a tool, but in their case you cannot pass additional arguments to the function that are not meant to be sent by the LLM. For example, as you will see later I'll use the `bigquery.Client`. For smolagents to use an external object from within the tool they have to make it global. Here is an example of how they do it in the [search_item_ctrl](https://github.com/huggingface/smolagents/blob/952d88f749265c9dce2fad46b2fe104bc60ce36a/src/smolagents/vision_web_browser.py#L91C5-L91C11) tool with its [global](https://github.com/huggingface/smolagents/blob/952d88f749265c9dce2fad46b2fe104bc60ce36a/src/smolagents/vision_web_browser.py#L203).

With my abstraction I can define extra `call_args` that will be passed to the function when the tool is used.

```python

def query_bigquery(inputs: BigtableQueryCommand, client: bigquery.Client):
   ...


tool = Tool(query_bigquery, call_args={"bigquery_client":client})
```

Once we have our list of tools, we need a way to run them when the LLM asks to. The LLM will return the name of the tool and a dictionary with the inputs.
For this I use a `ToolContainer` which is initialized with a list of `Tool`s and exposes two functions: `run_tool` and `claude_format`. The `run_tool` function gets the tool name and dictionary of inputs and runs the correct function, and `claude_format` will return a dictionary with the tools dictionary expected in the Claude client create method.

```python
class ToolsContainer:
    def __init__(self, tools: list[Tool]):
        self.tooldict = {t.name: t for t in tools}

    def run_tool(self, tool_name, inputs):
        try:
            tool_instance = self.tooldict[tool_name]
            if tool_instance.input_model is not None:
                inputs = tool_instance.input_model(**inputs)
                result = tool_instance.function(inputs=inputs, **tool_instance.additional_args)
            else:
                result = tool_instance.function(**tool_instance.additional_args)
        except Exception as e:
            result = e
        return str(result)

    def claude_format(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tooldict.values()
        ]
```

The final piece is the main agent loop. The `agentic_steps` calls the LLM and runs the tools until there are no more tool_use requests. To return the intermediate steps during the loop iterations some frameworks make this function a python generator like [agno](https://docs.agno.com/agents/run). I prefer passing a callback the function will call every iteration.

```python
def agentic_steps(
    messages: list[dict],
    claude_client: Anthropic,
    tools: ToolsContainer,
    system_prompt: str,
    callback: Callable,
    model: str,
    max_steps: int = float("inf"),
):
    while max_steps > 0:
        max_steps -= 1
        response = claude_client.messages.create(
            model=model,
            max_tokens=8192,
            tools=tools.claude_format(),
            system=system_prompt,
            messages=messages,
            temperature=0.0,
        )
        response_message = {"role": "assistant", "content": [c.model_dump() for c in response.content]}
        messages.append(response_message)

        if response.stop_reason == "tool_use":
            callback(response_message)
            parallel_tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_result = tools.run_tool(content.name, content.input)
                    parallel_tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result,
                        }
                    )
            new_message = {"role": "user", "content": parallel_tool_results}
            callback(new_message)
            messages.append(new_message)

        else:
            return response_message["content"][0]["text"]
    return "Reached max steps"
```

Finally, I wrap it all with a `TinyAgent` abstraction that just gets a text instruction, runs the agentic_steps and keeps track of the memory (message list).

```python
class TinyAgent:
    def __init__(
        self,
        claude_client: Anthropic,
        tools: ToolsContainer,
        system_prompt: str,
        callback: Callable,
        model: str,
    ) -> None:
        self.claude_client = claude_client
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model
        self.memory: list[dict] = []
        self.callback = callback

    def run(
        self,
        prompt: str,
        reset_messages: bool = False,
        max_steps: float = float("inf"),
    ) -> str:
        if reset_messages:
            self.memory = []
        self.memory.append({"role": "user", "content": prompt})
        final_response = agentic_steps(
            self.memory,
            self.claude_client,
            self.tools,
            self.system_prompt,
            self.callback,
            self.model,
            max_steps,
        )
        return final_response
```

That's it with the boilerplate, now let's make stuff.

### A Minimal Agent
I have a nice little abstraction, now I'll make a minimal agent using it. My agent will have two tools:  
**search_web** - A tool to search the web using DuckDuckGo API.

```python
class SearchWebCommand(BaseModel):
    query: str = Field(..., description="The query to search the web for")
    search_type: Literal["text", "news"] = Field(..., description="The type of search")
    max_results: int = Field(..., description="The maximum number of results")


def search_web(inputs: SearchWebCommand):
    """Search the web using DuckDuckGo."""
    if inputs.search_type == "text":
        return str(DDGS().text(inputs.query, max_results=7))
    elif inputs.search_type == "news":
        return str(DDGS().news(inputs.query, max_results=7))
    else:
        raise ValueError(f"Invalid search type: {inputs.search_type}")
```

**fetch_web_page** - This tool fetches a web page. This tool uses a very important recurring pattern I have noticed: instead of just returning the web page, the tool will extract only the relevant information from the web page. Tools that return full web pages or full results from anywhere are wasteful and spam the global message list. In the tool's inputs, the agent must specify the high-level goal of requesting this web page and the expected learnings when reading this page. The tool itself downloads the web page and calls another (smaller) LLM just to extract the relevant information from it.


```python
class FetchWebPageCommand(BaseModel):
    url: str = Field(..., description="The URL to fetch the web page from")
    required_learnings: str = Field(..., description="The required learnings to extract from the web page")
    high_level_goal: str = Field(..., description="The high level goal of why you need this web page")


def fetch_web_page(inputs: FetchWebPageCommand, claude_client: Anthropic):
    """Fetch the web page from the given URL."""
    page_markdown = md(requests.get(inputs.url).text)
    learning_prompt = dedent(
        f"""
        Given a high level goal of a user request about the web page, and the web page itself,
        return the expected learnings from this web page.
        Return any relevant information that can help the user to achieve the high level goal.
        Be concise, return only the learnings you are requested.

        The high level goal is:
        {inputs.high_level_goal}
        The required learnings are:
        {inputs.required_learnings}
        The web page is:
        {page_markdown}
        """
    )
    response = claude_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8192,
        system="",
        messages=[{"role": "user", "content": learning_prompt}],
        temperature=0.0,
    )
    return response.content[0].text
```
That's about it. I wrap it in a loop and print intermediate tool requests and results in [agent_template.py](https://github.com/yonigottesman/mimic-agent/blob/main/app/agent_template.py). Here is an example output of a single run:

<details style="width: 100%;  position: relative;">
<summary>expand</summary>
<div markdown="1">
```
> what is the weather like in Israel at May?
╭──────────────────────────────────────────────────────────────────────────────── Reasoning ────────────────────────────────────────────────────────────────────────────────╮
│ To answer your question about the weather in Israel during May, I'll need to search for some current information. Let me do that for you.                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Tool Use ─────────────────────────────────────────────────────────────────────────────────╮
│ Using tool: search_web with input:                                                                                                                                        │
│ max_results: 5                                                                                                                                                            │
│ query: weather in Israel during May climate temperature                                                                                                                   │
│ search_type: text                                                                                                                                                         │
│                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────────────────────────────────────── Tool Result ───────────────────────────────────────────────────────────────────────────────╮
│ [{'title': 'Weather in May in Israel - World Weather & Climate Information', 'href': 'https://weather-and-climate.com/averages-Israel-May', 'body': 'What is the weather  │
│ like in Israel in May? May in Israel generally has temperatures that range from warm to very warm, with precipitation levels ranging from almost none to low.             │
│ Temperatures The average maximum daytime temperatures vary from warm in Liman at 25°C to very warm in H̱aẕeva at 34°C.'}, {'title': 'Israel - temperatures in May -        │
│ Climates to Travel', 'href': 'https://www.climatestotravel.com/temperature/israel/may', 'body': 'May in Israel is on average a warm to hot month, with a minimum          │
│ temperature of 17.3 °C, a maximum of 29.3 °C, and therefore a daily average of 23.3 °C. The minimum temperature is normally recorded before dawn, the maximum in the      │
│ early afternoon.'}, {'title': 'May weather - Spring 2025 - Israel', 'href': 'https://www.weather-atlas.com/en/israel-weather-may', 'body': 'With rainfall becoming a rare │
│ occurrence, the weather becomes distinctly warmer. Average temperatures across the country during May generally range from 18°C (64.4°F) to 30°C (86°F). Jerusalem,       │
│ situated in the central region, experiences temperatures typically ranging from 16°C (60.8°F) to 28°C (82.4°F).'}, {'title': 'Israel climate: average weather,            │
│ temperature, rain - Climates to Travel', 'href': 'https://www.climatestotravel.com/climate/israel', 'body': 'Israel, a small country in western Asia, facing the          │
│ Mediterranean to the west and the Red Sea to the south, has a Mediterranean climate on the west coast, temperate in the hills of the interior, and desert in the flat     │
│ areas of the east and the south. The climate in Israel is characterized by hot, sunny summers and by a long period of sunshine, with clear skies from May to              │
│ September.'}, {'title': 'Israel Weather in May 2025: Perfectly Warm and Dry - Global Highlights', 'href': 'https://www.globalhighlights.com/israel/weather-in-may',       │
│ 'body': '6. May Events; Israel Weather in May Overview. Average temperature range: 19°C-29°C (67°F-85°F) Season: Warm and dry; Rainfall: 0 mm (0 inches) throughout the   │
│ country; Rainy days: 1 in Jerusalem and 2 in Tel Aviv; May Weather by Region in Israel. Israel is divided into four separate climate regions.'}, {'title': 'Israel may    │
│ Weather 2025: Forecast and temperature city by city', 'href': 'https://www.whereandwhen.net/when/middle-east/israel/may/', 'body': 'Climate in Israel in may Climate in   │
│ Jerusalem in may. In the month of may, maximum temperature is 85°F and minimum temperature is 72°F (for an average temperature of 79°F). The climate quite comfortable in │
│ this city in may. does not rain (or really exceptionally) during the month of may.'}, {'title': 'May weather - Spring 2025 - Jerusalem, Israel', 'href':                  │
│ 'https://www.weather-atlas.com/en/israel/jerusalem-weather-may', 'body': "The high-temperature in Jerusalem, Israel, during May is 29.1°C (84.4°F) and the                │
│ low-temperature is 18.4°C (65.1°F).In May, the average heat index (a.k.a. 'real feel', 'feels like'), which calculates the temperature value influenced by the moisture   │
│ in the air, is calculated to be 29°C (84.2°F)."}]                                                                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Reasoning ────────────────────────────────────────────────────────────────────────────────╮
│ Let me get more specific information about the weather in different parts of Israel during May:                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Tool Use ─────────────────────────────────────────────────────────────────────────────────╮
│ Using tool: fetch_web_page with input:                                                                                                                                    │
│ high_level_goal: To provide accurate information about the weather in Israel during                                                                                       │
│   May                                                                                                                                                                     │
│ required_learnings: Detailed temperature information for different regions of Israel                                                                                      │
│   in May, rainfall patterns, and general weather conditions                                                                                                               │
│ url: https://www.climatestotravel.com/temperature/israel/may                                                                                                              │
│                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────────────────────────────────────── Tool Result ───────────────────────────────────────────────────────────────────────────────╮
│ Based on the web page, here are the key learnings about Israel's weather in May:                                                                                          │
│                                                                                                                                                                           │
│ Temperature Information:                                                                                                                                                  │
│ - Average minimum temperature: 17.3°C                                                                                                                                     │
│ - Average maximum temperature: 29.3°C                                                                                                                                     │
│ - Daily average temperature: 23.3°C                                                                                                                                       │
│                                                                                                                                                                           │
│ Regional Temperature Variations:                                                                                                                                          │
│ - Coolest region: Safed (14.7°C min, 25.2°C max)                                                                                                                          │
│ - Hottest region: Eilat (22.7°C min, 36.1°C max)                                                                                                                          │
│                                                                                                                                                                           │
│ Temperature by Major Cities:                                                                                                                                              │
│ - Tel Aviv: 16.4°C min, 27.7°C max                                                                                                                                        │
│ - Jerusalem: 16.2°C min, 26.2°C max                                                                                                                                       │
│ - Haifa: 18.6°C min, 26.3°C max                                                                                                                                           │
│ - Tiberias: 19.1°C min, 33.2°C max                                                                                                                                        │
│                                                                                                                                                                           │
│ General Observation:                                                                                                                                                      │
│ - May is considered a warm to hot month in Israel                                                                                                                         │
│ - Temperatures vary significantly depending on the region's elevation and location                                                                                        │
│                                                                                                                                                                           │
│ Note: The webpage does not provide specific information about rainfall patterns in May.                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────── Tool Use ─────────────────────────────────────────────────────────────────────────────────╮
│ Using tool: fetch_web_page with input:                                                                                                                                    │
│ high_level_goal: To provide comprehensive information about the weather in Israel                                                                                         │
│   during May                                                                                                                                                              │
│ required_learnings: Information about rainfall, humidity, and regional weather differences                                                                                │
│   in Israel during May                                                                                                                                                    │
│ url: https://www.globalhighlights.com/israel/weather-in-may                                                                                                               │
│                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────────────────────────────────────── Tool Result ───────────────────────────────────────────────────────────────────────────────╮
│ Based on the web page, here are the key learnings about Israel's weather in May:                                                                                          │
│                                                                                                                                                                           │
│ Rainfall:                                                                                                                                                                 │
│ - Essentially no rainfall across Israel in May                                                                                                                            │
│ - 0 mm (0 inches) of rain throughout the country                                                                                                                          │
│ - Only 1 rainy day in Jerusalem and 2 in Tel Aviv                                                                                                                         │
│                                                                                                                                                                           │
│ Humidity and Temperature:                                                                                                                                                 │
│ - Average temperature range: 19°C-29°C (67°F-85°F)                                                                                                                        │
│ - Warm and dry season                                                                                                                                                     │
│ - Regional temperature variations:                                                                                                                                        │
│   - Northern Hills/Tiberias: 19°C-33°C (66°F-91°F)                                                                                                                        │
│   - Tel Aviv/Haifa: 17°C-25°C (63°F-77°F)                                                                                                                                 │
│   - Jerusalem: 16°C-25°C (61°F-77°F)                                                                                                                                      │
│   - Southern Desert/Dead Sea: 25°C-34°C (77°F-93°F)                                                                                                                       │
│                                                                                                                                                                           │
│ Regional Weather Differences:                                                                                                                                             │
│ - Northern regions tend to be hotter                                                                                                                                      │
│ - Coastal cities like Tel Aviv are slightly cooler                                                                                                                        │
│ - Jerusalem is relatively cooler compared to other regions                                                                                                                │
│ - Southern desert areas are extremely hot and dry                                                                                                                         │
│                                                                                                                                                                           │
│ The page emphasizes that May is warm, sunny, and essentially rainless across all regions of Israel.                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────────────── Final Response ──────────────────────────────────────────────────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃                                                                     Weather in Israel during May                                                                      ┃ │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │
│                                                                                                                                                                           │
│ Based on the information I've gathered, here's what the weather is like in Israel during May:                                                                             │
│                                                                                                                                                                           │
│                                                                                                                                                                           │
│                                                                              General Weather                                                                              │
│                                                                                                                                                                           │
│ May in Israel is characterized by warm to hot temperatures and very dry conditions. It marks the beginning of the summer season with clear skies and abundant sunshine.   │
│ Rainfall is extremely rare during this month.                                                                                                                             │
│                                                                                                                                                                           │
│                                                                                                                                                                           │
│                                                                                Temperature                                                                                │
│                                                                                                                                                                           │
│ The average temperature range across Israel in May is 19°C-29°C (67°F-85°F), but there are significant regional variations:                                               │
│                                                                                                                                                                           │
│  • Coastal Areas (Tel Aviv, Haifa): Milder temperatures ranging from 16-18°C (61-64°F) at night to 25-28°C (77-82°F) during the day. The Mediterranean Sea has a          │
│    moderating effect on these areas.                                                                                                                                      │
│  • Jerusalem and Central Hills: Slightly cooler with temperatures typically ranging from 16°C (61°F) at night to 25-26°C (77-79°F) during the day.                        │
│  • Northern Hills/Tiberias: Warmer with temperatures ranging from 19°C (66°F) to 33°C (91°F).                                                                             │
│  • Southern Desert/Dead Sea/Eilat: The hottest regions with temperatures ranging from 22-25°C (72-77°F) at night to 34-36°C (93-97°F) during the day.                     │
│                                                                                                                                                                           │
│                                                                                                                                                                           │
│                                                                                 Rainfall                                                                                  │
│                                                                                                                                                                           │
│ May is essentially rainless throughout Israel:                                                                                                                            │
│                                                                                                                                                                           │
│  • 0 mm (0 inches) of precipitation is typical                                                                                                                            │
│  • Jerusalem might experience just 1 rainy day                                                                                                                            │
│  • Tel Aviv might have 2 rainy days, but actual rainfall is minimal                                                                                                       │
│                                                                                                                                                                           │
│                                                                                                                                                                           │
│                                                                                 Humidity                                                                                  │
│                                                                                                                                                                           │
│  • Coastal areas tend to be more humid due to proximity to the Mediterranean                                                                                              │
│  • Interior and desert regions are very dry                                                                                                                               │
│                                                                                                                                                                           │
│                                                                                                                                                                           │
│                                                                                  Overall                                                                                  │
│                                                                                                                                                                           │
│ May is considered an excellent time to visit Israel weather-wise, as it's warm and sunny but not yet experiencing the extreme heat of summer (particularly June-August).  │
│ The weather is generally pleasant for outdoor activities, though it can get quite hot in the desert regions.                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
</div>
</details>


## Mimic-III SQL Agent
{% include mermaid.html %}

<div class="mermaid">
flowchart TD
    User([User])
    Agent([Agent])
    BQ[Query BigQuery]
    TD[Get Table Description and Metadata]
    Retrieval[Retrieve Similar Example Queries]

    %% Main flow
    User -->|"user query"| Agent
    Agent -->|"result"| User
    
    %% Circular connections
    Agent <-->|"execute queries"| BQ
    Agent <-->|"retrieve metadata"| TD
    Agent <-->|"search examples"| Retrieval

    %% Styling with better dark/light mode compatibility
    classDef user fill:#7b68ee,stroke:#483d8b,color:#ffffff,stroke-width:1px
    classDef agent fill:#ff7f50,stroke:#ff4500,color:#ffffff,stroke-width:1px
    classDef process fill:#3cb371,stroke:#2e8b57,color:#ffffff,stroke-width:1px
    
    class User user
    class Agent agent
    class BQ,TD,Retrieval process
</div>


The [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) database is a comprehensive critical care dataset with tables for patients, admissions, vital signs, lab tests, medications, procedures, and clinical notes.

I chose MIMIC-III, but the techniques and concepts I'll show are valid for any database. Exploring this data requires understanding the structure of the tables, their relationships, and what each table contains. While the database is fairly well [documented](https://mimic.mit.edu/docs/iii/), it's still challenging to formulate complex SQL statements for specific business use cases. That's where the MIMIC-III agent comes in. The agent will process high-level user requests and interact with various tools to explore the database.

To access the database, I followed these [instructions](https://mimic.mit.edu/docs/iii/tutorials/intro-to-mimic-iii-bq/) to load it into BigQuery, but you can download the tables and load them into any database.

### The System Prompt
I don't want a huge system prompt with all the table information, but I do want the agent to have an initial high-level understanding of the database and its various tables.

The MIMIC [website](https://mimic.mit.edu/docs/iii/tables/) has a [repo](https://github.com/MIT-LCP/mimic-website/tree/main/content/en/docs/III/tables) containing detailed documentation for each table. For the system prompt, I use the `Table purpose` section from each table's documentation. This will serve as a good starting point for the agent when exploring different tables.

My system prompt is:

```
* You are a helpful assistant that accesses the MIMIC-III database to answer questions.
* Database locations:
  - Clinical data: `mimiciii_clinical` (without the `physionet-data` prefix)
  - Notes data: `mimiciii_notes` (without the `physionet-data` prefix)
* The query syntax is BIGQUERY SQL!
* Always start by requesting similar example queries.
* Process for answering:
  1. Identify relevant tables for the question
  2. Check table schemas before querying
  3. Extract information from appropriate tables
  4. Provide concise, direct answers that address EXACTLY what was asked
  5. If in doubt, ask for more information
* Keep conversations short! Try to do at most 1 query per user request!!
* The only available tables are:
{get_highlevel_tables_information()}
```

## Tools
**get_table_schema_and_description** - This tool is used by the agent to examine a table more deeply. The tool will read the table documentation and extract any relevant information the agent might need. It follows the same pattern as the `fetch_web_page` tool from before: instead of just dumping the full table documentation and column schema, it will call a smaller LLM that will extract only the information relevant to the agent's high-level goal and expected learnings.

```python
class GetTableSchemaAndDescriptionInput(BaseModel):
    table_name: Table = Field(description="mimic-iii table name")
    high_level_goal: str = Field(description="high level goal of why you need this table")
    expected_learnings: str = Field(
        description="expected learnings from this tool. You can ask to learn about columns schemas for example"
    )


class GetTableSchemaAndDescriptionInputBatch(BaseModel):
    table_description_requests: list[GetTableSchemaAndDescriptionInput] = Field(
        description="list of table description requests"
    )


def get_table_schema_and_description(inputs: GetTableSchemaAndDescriptionInputBatch, claude_client: Anthropic) -> str:
    """Get information about a mimic-iii table. This tool will not query the table,
    it will only provide information about the table."""

    responses = []

    for input in inputs.table_description_requests:
        path = (Path("app/resources/tables") / input.table_name.value.lower()).with_suffix(".md")
        text = path.read_text()

        learning_prompt = dedent(
            f"""
            Given a high level goal of a user request about the mimic-iii tables, and a detailed description of a table,
            return the expected learnings from this table.
            Return any relevant information that ca help the user to achieve the high level goal.
            For exaple only return the columns that are needed to achieve the high level goal.
            Be concise, return only the learnings you are requested.

            The high level goal is:
            {input.high_level_goal}
            The required learnings are:
            {input.expected_learnings}
            The table name is:
            {input.table_name}
            The table description is:
            {text}
            """
        )
        response = claude_client.messages.create(
            model="claude-3-5-haiku-20241022", 
            max_tokens=8192,
            system="",
            messages=[{"role": "user", "content": learning_prompt}],
            temperature=0.0,
        )
        responses.append(f"Table: {input.table_name.value}\n{response.content[0].text}")
    return "\n\n".join(responses)
```
**find_similar_queries** - The [mimic-code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts) repo contains the descriptions and SQL statements of all derived MIMIC-III tables.
For example, [norepinephrine_durations.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/norepinephrine_durations.sql) shows the "Start and stop times for administration of norepinephrine" (description taken from README). There is also a [cookbook](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts/cookbook) with numerous SQL queries and descriptions, such as [age_histogram.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/cookbook/age_histogram.sql). Each query begins with a short descriptive comment.
I'm going to index all these descriptions into a [LanceDB](https://github.com/lancedb/lancedb) database and the tool will perform hybrid search to retrieve the 3 most similar queries.

Indexing is relatively straightforward and done in [index_query_examples](https://github.com/yonigottesman/mimic-agent/blob/main/app/index_query_examples.py). I use the `Alibaba-NLP/gte-modernbert-base` embedding model, and I don't need any chunking strategies as the example queries are very short in text.  
Once the index is ready and saved, the `find_similar_queries` tool can retrieve the most relevant queries.

```python
class FindRelevantExampleQueriesInput(BaseModel):
    query_description: str = Field(description="Concise description of the query to find similar queries for")


def find_similar_queries(
    inputs: FindRelevantExampleQueriesInput,
    lancedb_table: lancedb.table.LanceTable,
    model: SentenceTransformer,
) -> str:
    """Given a query description, retrieve similar query descriptions and the sql queries themselves"""
    query_embedding = model.encode(inputs.query_description, convert_to_tensor=False)

    result = (
        lancedb_table.search(
            query_type="hybrid",
            vector_column_name="query_description_vector",
            fts_columns="query_description",
        )
        .text(inputs.query_description)
        .vector(query_embedding)
        .bypass_vector_index()  # exhaustive search
        .limit(10)
    )
    df = result.to_df()
    indices = maximal_marginal_relevance(query_embedding, df["query_description_vector"].tolist(), k=3)
    df = df.iloc[indices]
    return "\n".join(
        df[["query_description", "query_sql"]]
        .apply(lambda r: f"-- {r.query_description}\n{r.query_sql}", axis=1)
        .tolist()
    )
```

The examples and cookbook contain multiple SQL queries that essentially do the same thing for different cases. For example, part of the `duration` concepts perform identical operations but for different medications, as seen in [dobutamine_dose.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/dobutamine_dose.sql) and [dopamine_dose.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/dopamine_dose.sql). Returning both these examples doesn't add any more information for the agent, and it's better if I only return one.
To overcome this issue, I use `maximal_marginal_relevance` for reranking in order to get `k=3` diverse examples.

**query_db** - The main tool to interact with BigQuery. It receives an SQL statement and executes it.

```python
class QueryDBInput(BaseModel):
    query: str = Field(description="bigquery sql query to execute. Always limit the result to max 10 rows.")


def query_db(inputs: QueryDBInput, client: bigquery.Client) -> str:
    """Query the mimic-iii bigquery database. Will return only first 10 rows."""
    query_job = client.query(inputs.query)
    df = query_job.to_dataframe()
    df = df.map(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
    return yaml.dump(df.head(10).to_dict(orient="records"))
```

I wrap the TinyAgent in a [streamlit app](https://github.com/yonigottesman/mimic-agent/blob/main/app/main.py) and deployed it to [https://mimic-agent.yonigo.co/](https://mimic-agent.yonigo.co/) using gcloud Cloud Run.


### Example Run

For the user query 'Find the most commonly prescribed medications for heart failure patients', here is the output of a real agent run. You can follow the steps and see how the agent tries different tables, searches for similar queries, and requests schema and information about the different tables.

<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">User Query</summary>
  <div markdown="1">
  Find the most commonly prescribed medications for heart failure patients
  </div>
</details>

<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
I'll help you find the most commonly prescribed medications for heart failure patients in the MIMIC-III database. To do this, I need to understand which tables contain information about medications and diagnoses, and then query them appropriately.

First, let me explore the relevant tables:

Using tool: **get_table_schema_and_description** with input:

```
table_description_requests:
- expected_learnings: Learn about the structure of the prescriptions table to identify
    medication names and frequencies
  high_level_goal: Find medication information for heart failure patients
  table_name: prescriptions
- expected_learnings: Learn how diagnoses are stored and how to identify heart failure
    diagnoses
  high_level_goal: Identify heart failure patients
  table_name: diagnoses_icd
- expected_learnings: Learn how to map ICD codes to heart failure diagnoses
  high_level_goal: Find ICD codes for heart failure
  table_name: d_icd_diagnoses
```
  </div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">
Table: prescriptions
For the goal of finding medication information for heart failure patients, here are the key learnings from the PRESCRIPTIONS table:

Relevant Columns:
- SUBJECT_ID: To link medications to specific patients
- DRUG_NAME_GENERIC: To identify specific medication names
- DRUG_TYPE: To categorize medications
- STARTDATE, ENDDATE: To track medication prescription periods
- DOSE_VAL_RX, DOSE_UNIT_RX: To understand medication dosage
- ROUTE: To understand how the medication was administered

Key Insights:
1. You can track heart failure patient medications by filtering SUBJECT_IDs from heart failure patients
2. Multiple medication representations available (generic name, POE name)
3. Can analyze medication frequency, duration, and dosage
4. Provides comprehensive medication prescription details

Recommended Analysis Approach:
- Join with PATIENTS or DIAGNOSES_ICD tables to identify heart failure patients
- Use DRUG_NAME_GENERIC to categorize heart failure related medications
- Analyze prescription patterns, dosages, and routes of administration

Table: diagnoses_icd
Based on the high-level goal of identifying heart failure patients and the need to learn how diagnoses are stored, here are the key learnings from the DIAGNOSES_ICD table:

Relevant Columns:
- SUBJECT_ID: to track individual patients
- HADM_ID: to track specific hospital admissions
- ICD9_CODE: to identify specific diagnoses

Key Insights for Identifying Heart Failure:
1. Heart failure diagnoses will be stored as specific ICD-9 codes
2. You'll need to look up the exact ICD-9 codes for heart failure (e.g., 428.xx series)
3. The SEQ_NUM can help prioritize principal vs. secondary diagnoses
4. Codes are generated at the end of hospital stay for billing purposes

Recommended Next Steps:
- Cross-reference with D_ICD_DIAGNOSES table to get full diagnosis descriptions
- Use ICD9_CODE to filter and identify heart failure patients
- Consider both principal and secondary diagnoses using SEQ_NUM

Note: Ensure you're using the correct ICD-9 codes for heart failure when querying this table.

Table: d_icd_diagnoses
Based on the high-level goal of finding ICD codes for heart failure and the need to map ICD codes to heart failure diagnoses, here are the key learnings from the D_ICD_DIAGNOSES table:

Relevant Columns:
- ICD9_CODE: The specific code to identify heart failure diagnoses
- SHORT_TITLE: Quick reference for the diagnosis
- LONG_TITLE: Detailed description of the diagnosis

Recommended Approach:
1. Search for ICD9_CODE entries with heart failure-related titles
2. Use the SHORT_TITLE and LONG_TITLE to confirm the specific heart failure diagnosis
3. Collect the relevant ICD9_CODE values for further analysis

Potential Heart Failure ICD-9 Codes to Look For:
- Codes likely starting with 428.xx (Congestive Heart Failure)
- Codes related to specific types of heart failure

Next Steps:
- Cross-reference these codes with the DIAGNOSES_ICD table to find patient diagnoses
- Validate the codes match your specific heart failure research criteria
</div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
Now, let me find similar queries that might help us structure our query correctly:

Using tool: **find_similar_queries** with input:
```
query_description: Most commonly prescribed medications for heart failure patients
```
</div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">
```sql
-- Dose administered with start/stop times for norepinephrine.
-- This query extracts dose+durations of norepinephrine administration
-- Total time on the drug can be calculated from this table by grouping using ICUSTAY_ID

-- Get drug administration data from CareVue first
with vasocv1 as
(
  select
    cv.icustay_id, cv.charttime
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    , max(case when itemid in (30047,30120) then 1 else 0 end) as vaso -- norepinephrine

    -- the 'stopped' column indicates if a vasopressor has been disconnected
    , max(case when itemid in (30047,30120) and (stopped = 'Stopped' OR stopped like 'D/C%') then 1
          else 0 end) as vaso_stopped

  -- case statement determining whether the ITEMID is an instance of vasopressor usage

    , max(case when itemid in (30047,30120) and rate is not null then 1 else 0 end) as vaso_null
    , max(case
            when itemid = 30047 and wd.weight is null then rate / 80.0 -- this is rare, only affects a total of ~400 rows
            when itemid = 30047 then rate / wd.weight -- measured in mcgmin
            when itemid = 30120 then rate -- measured in mcgkgmin ** there are clear errors, perhaps actually mcgmin
          else null end) as vaso_rate
    , max(case when itemid in (30047,30120) then amount else null end) as vaso_amount

  FROM `physionet-data.mimiciii_clinical.inputevents_cv` cv
  left join `physionet-data.mimiciii_derived.weight_durations` wd
    on cv.icustay_id = wd.icustay_id
    and cv.charttime between wd.starttime and wd.endtime
  where itemid in (30047,30120) -- norepinephrine
  and cv.icustay_id is not null
  group by cv.icustay_id, cv.charttime
)
, vasocv2 as
(
  select v.*
    , sum(vaso_null) over (partition by icustay_id order by charttime) as vaso_partition
  from
    vasocv1 v
)
, vasocv3 as
(
  select v.*
    , first_value(vaso_rate) over (partition by icustay_id, vaso_partition order by charttime) as vaso_prevrate_ifnull
  from
    vasocv2 v
)
, vasocv4 as
(
select
    icustay_id
    , charttime
    -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta

    , vaso
    , vaso_rate
    , vaso_amount
    , vaso_stopped
    , vaso_prevrate_ifnull

    -- We define start time here
    , case
        when vaso = 0 then null

        -- if this is the first instance of the vasoactive drug
        when vaso_rate > 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, vaso, vaso_null
          order by charttime
          )
          is null
          then 1

        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
        when vaso_rate = 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, vaso
          order by charttime
          )
          = 0
          then 0

        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
        when vaso_prevrate_ifnull = 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, vaso
          order by charttime
          )
          = 0
          then 0

        -- If the last recorded rate was 0, newvaso = 1
        when LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, vaso
          order by charttime
          ) = 0
          then 1

        -- If the last recorded vaso was D/C'd, newvaso = 1
        when
          LAG(vaso_stopped,1)
          OVER
          (
          partition by icustay_id, vaso
          order by charttime
          )
          = 1 then 1

        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
      else null
      end as vaso_start

FROM
  vasocv3
)
-- propagate start/stop flags forward in time
, vasocv5 as
(
  select v.*
    , SUM(vaso_start) OVER (partition by icustay_id, vaso order by charttime) as vaso_first
FROM
  vasocv4 v
)
, vasocv6 as
(
  select v.*
    -- We define end time here
    , case
        when vaso = 0
          then null

        -- If the recorded vaso was D/C'd, this is an end time
        when vaso_stopped = 1
          then vaso_first

        -- If the rate is zero, this is the end time
        when vaso_rate = 0
          then vaso_first

        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on vasopressors
        -- in principle, this could add an extra end time for the vasopressor
        -- however, since we later group on vaso_start, any extra end times are ignored
        when LEAD(CHARTTIME,1)
          OVER
          (
          partition by icustay_id, vaso
          order by charttime
          ) is null
          then vaso_first

        else null
        end as vaso_stop
    from vasocv5 v
)

-- -- if you want to look at the results of the table before grouping:
-- select
--   icustay_id, charttime, vaso, vaso_rate, vaso_amount
--     , vaso_stopped
--     , vaso_start
--     , vaso_first
--     , vaso_stop
-- from vasocv6 order by icustay_id, charttime;

, vasocv7 as
(
select
  icustay_id
  , charttime as starttime
  , lead(charttime) OVER (partition by icustay_id, vaso_first order by charttime) as endtime
  , vaso, vaso_rate, vaso_amount, vaso_stop, vaso_start, vaso_first
from vasocv6
where
  vaso_first is not null -- bogus data
and
  vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
and
  icustay_id is not null -- there are data for "floating" admissions, we don't worry about these
)
-- table of start/stop times for event
, vasocv8 as
(
  select
    icustay_id
    , starttime, endtime
    , vaso, vaso_rate, vaso_amount, vaso_stop, vaso_start, vaso_first
  from vasocv7
  where endtime is not null
  and vaso_rate > 0
  and starttime != endtime
)
-- collapse these start/stop times down if the rate doesn't change
, vasocv9 as
(
  select
    icustay_id
    , starttime, endtime
    , case
        when LAG(endtime) OVER (partition by icustay_id order by starttime, endtime) = starttime
        AND  LAG(vaso_rate) OVER (partition by icustay_id order by starttime, endtime) = vaso_rate
        THEN 0
      else 1
    end as vaso_groups
    , vaso, vaso_rate, vaso_amount, vaso_stop, vaso_start, vaso_first
  from vasocv8
  where endtime is not null
  and vaso_rate > 0
  and starttime != endtime
)
, vasocv10 as
(
  select
    icustay_id
    , starttime, endtime
    , vaso_groups
    , SUM(vaso_groups) OVER (partition by icustay_id order by starttime, endtime) as vaso_groups_sum
    , vaso, vaso_rate, vaso_amount, vaso_stop, vaso_start, vaso_first
  from vasocv9
)
, vasocv as
(
  select icustay_id
  , min(starttime) as starttime
  , max(endtime) as endtime
  , vaso_groups_sum
  , vaso_rate
  , sum(vaso_amount) as vaso_amount
  from vasocv10
  group by icustay_id, vaso_groups_sum, vaso_rate
)
-- now we extract the associated data for metavision patients
, vasomv as
(
  select
    icustay_id, linkorderid
    , rate as vaso_rate
    , amount as vaso_amount
    , starttime
    , endtime
  from `physionet-data.mimiciii_clinical.inputevents_mv`
  where itemid = 221906 -- norepinephrine
  and statusdescription != 'Rewritten' -- only valid orders
)
-- now assign this data to every hour of the patient's stay
-- vaso_amount for carevue is not accurate
SELECT icustay_id
  , starttime, endtime
  , vaso_rate, vaso_amount
from vasocv
UNION ALL
SELECT icustay_id
  , starttime, endtime
  , vaso_rate, vaso_amount
from vasomv
order by icustay_id, starttime;

-- Retrieves the systolic blood pressure for adult patients
-- --------------------------------------------------------
-- Title: Retrieves the systolic blood pressure for adult patients
-- Notes: this query does not specify a schema. To run it on your local
-- MIMIC schema, run the following command:
--  SET SEARCH_PATH TO mimiciii;
-- Where "mimiciii" is the name of your schema, and may be different.
-- --------------------------------------------------------

WITH agetbl AS
(
  SELECT ad.subject_id
  FROM `physionet-data.mimiciii_clinical.admissions` ad
  INNER JOIN patients p
  ON ad.subject_id = p.subject_id
  WHERE
  -- filter to only adults
  DATETIME_DIFF(ad.admittime, p.dob, YEAR) > 15
  -- group by subject_id to ensure there is only 1 subject_id per row
  group by ad.subject_id
)
, sysbp as
(
  SELECT width_bucket(valuenum, 0, 300, 300) AS bucket
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  INNER JOIN agetbl
  ON ce.subject_id = agetbl.subject_id
  WHERE itemid IN
  (
      6 -- ABP [Systolic]
    , 51 -- Arterial BP [Systolic]
    , 455 -- NBP [Systolic]
    , 6701 -- Arterial BP #2 [Systolic]
    , 220050 -- Arterial Blood Pressure systolic
    , 220179 -- Non Invasive Blood Pressure systolic
  )
)
SELECT bucket as systolic_blood_pressure, count(*)
FROM sysbp
GROUP BY bucket
ORDER BY bucket;

-- Stages of acute kidney failure (AKI) as defined by KDIGO.
-- This query checks if the patient had AKI according to KDIGO.
-- AKI is calculated every time a creatinine or urine output measurement occurs.
-- Baseline creatinine is defined as the lowest creatinine in the past 7 days.

-- get creatinine stages
with cr_stg AS
(
  SELECT
    cr.icustay_id
    , cr.charttime
    , cr.creat
    , case
        -- 3x baseline
        when cr.creat >= (cr.creat_low_past_7day*3.0) then 3
        -- *OR* cr >= 4.0 with associated increase
        when cr.creat >= 4
        -- For patients reaching Stage 3 by SCr >4.0 mg/dl
        -- require that the patient first achieve ... acute increase >= 0.3 within 48 hr
        -- *or* an increase of >= 1.5 times baseline
        and (cr.creat_low_past_48hr <= 3.7 OR cr.creat >= (1.5*cr.creat_low_past_7day))
            then 3 
        -- TODO: initiation of RRT
        when cr.creat >= (cr.creat_low_past_7day*2.0) then 2
        when cr.creat >= (cr.creat_low_past_48hr+0.3) then 1
        when cr.creat >= (cr.creat_low_past_7day*1.5) then 1
    else 0 end as aki_stage_creat
  FROM `physionet-data.mimiciii_derived.kdigo_creatinine` cr
)
-- stages for UO / creat
, uo_stg as
(
  select
      uo.icustay_id
    , uo.charttime
    , uo.weight
    , uo.uo_rt_6hr
    , uo.uo_rt_12hr
    , uo.uo_rt_24hr
    -- AKI stages according to urine output
    , CASE
        WHEN uo.uo_rt_6hr IS NULL THEN NULL
        -- require patient to be in ICU for at least 6 hours to stage UO
        WHEN uo.charttime <= DATETIME_ADD(ie.intime, INTERVAL '6' HOUR) THEN 0
        -- require the UO rate to be calculated over half the period
        -- i.e. for uo rate over 24 hours, require documentation at least 12 hr apart
        WHEN uo.uo_tm_24hr >= 11 AND uo.uo_rt_24hr < 0.3 THEN 3
        WHEN uo.uo_tm_12hr >= 5 AND uo.uo_rt_12hr = 0 THEN 3
        WHEN uo.uo_tm_12hr >= 5 AND uo.uo_rt_12hr < 0.5 THEN 2
        WHEN uo.uo_tm_6hr >= 2 AND uo.uo_rt_6hr  < 0.5 THEN 1
    ELSE 0 END AS aki_stage_uo
  from `physionet-data.mimiciii_derived.kdigo_uo` uo
  INNER JOIN `physionet-data.mimiciii_clinical.icustays` ie
    ON uo.icustay_id = ie.icustay_id
)
-- get all charttimes documented
, tm_stg AS
(
    SELECT
      icustay_id, charttime
    FROM cr_stg
    UNION DISTINCT
    SELECT
      icustay_id, charttime
    FROM uo_stg
)
select
    ie.icustay_id
  , tm.charttime
  , cr.creat
  , cr.aki_stage_creat
  , uo.uo_rt_6hr
  , uo.uo_rt_12hr
  , uo.uo_rt_24hr
  , uo.aki_stage_uo
  -- Classify AKI using both creatinine/urine output criteria
  , GREATEST(
      COALESCE(cr.aki_stage_creat, 0),
      COALESCE(uo.aki_stage_uo, 0)
    ) AS aki_stage
FROM `physionet-data.mimiciii_clinical.icustays` ie
-- get all possible charttimes as listed in tm_stg
LEFT JOIN tm_stg tm
  ON ie.icustay_id = tm.icustay_id
LEFT JOIN cr_stg cr
  ON ie.icustay_id = cr.icustay_id
  AND tm.charttime = cr.charttime
LEFT JOIN uo_stg uo
  ON ie.icustay_id = uo.icustay_id
  AND tm.charttime = uo.charttime
order by ie.icustay_id, tm.charttime;
```
</div>
</details>

<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
Now, let me look up the ICD-9 codes for heart failure to identify heart failure patients:

Using tool: **query_db** with input:
```sql
SELECT icd9_code, short_title, long_title 
FROM `physionet-data.mimiciii_clinical.d_icd_diagnoses` 
WHERE short_title LIKE '%heart failure%' OR long_title LIKE '%heart failure%'
LIMIT 10
```
</div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">
- icd9_code: '4280'
  long_title: Congestive heart failure, unspecified
  short_title: CHF NOS
- icd9_code: '4281'
  long_title: Left heart failure
  short_title: Left heart failure
- icd9_code: '40200'
  long_title: Malignant hypertensive heart disease without heart failure
  short_title: Mal hyp ht dis w/o hf
- icd9_code: '40211'
  long_title: Benign hypertensive heart disease with heart failure
  short_title: Benign hyp ht dis w hf
- icd9_code: '42832'
  long_title: Chronic diastolic heart failure
  short_title: Chr diastolic hrt fail
- icd9_code: '40290'
  long_title: Unspecified hypertensive heart disease without heart failure
  short_title: Hyp hrt dis NOS w/o hf
- icd9_code: '42823'
  long_title: Acute on chronic systolic heart failure
  short_title: Ac on chr syst hrt fail
- icd9_code: '42821'
  long_title: Acute systolic heart failure
  short_title: Ac systolic hrt failure
- icd9_code: '40411'
  long_title: Hypertensive heart and chronic kidney disease, benign, with heart failure
    and with chronic kidney disease stage I through stage IV, or unspecified
  short_title: Ben hyp ht/kd I-IV w hf
- icd9_code: '40491'
  long_title: Hypertensive heart and chronic kidney disease, unspecified, with heart
    failure and with chronic kidney disease stage I through stage IV, or unspecified
  short_title: Hyp ht/kd NOS I-IV w hf

</div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
Now I'll create a query to find the most commonly prescribed medications for heart failure patients:

Using tool: **query_db** with input:
```sql
WITH heart_failure_patients AS (
    SELECT DISTINCT p.subject_id
    FROM `physionet-data.mimiciii_clinical.patients` p
    INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d
    ON p.subject_id = d.subject_id
    WHERE d.icd9_code IN ('4280', '4281', '42821', '42823', '42832', '40211', '40411', '40491')
)

SELECT 
    pr.drug_name_generic,
    COUNT(*) as prescription_count
FROM `physionet-data.mimiciii_clinical.prescriptions` pr
INNER JOIN heart_failure_patients hf
ON pr.subject_id = hf.subject_id
WHERE pr.drug_name_generic IS NOT NULL
GROUP BY pr.drug_name_generic
ORDER BY prescription_count DESC
LIMIT 10
```
</div>
</details>

<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">
- drug_name_generic: Furosemide
  prescription_count: 67710
- drug_name_generic: Potassium Chloride
  prescription_count: 35366
- drug_name_generic: Metoprolol
  prescription_count: 31901
- drug_name_generic: Sodium Chloride 0.9%  Flush
  prescription_count: 26784
- drug_name_generic: Insulin - Sliding Scale
  prescription_count: 26224
- drug_name_generic: Metoprolol Tartrate
  prescription_count: 22769
- drug_name_generic: Heparin Sodium
  prescription_count: 20007
- drug_name_generic: Acetaminophen
  prescription_count: 19735
- drug_name_generic: Warfarin
  prescription_count: 19598
- drug_name_generic: Lorazepam
  prescription_count: 15899

</div>
</details>





<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
Let me also check the most common heart failure medications by looking at specific heart failure drug classes:

Using tool: **query_db** with input:

```sql
WITH heart_failure_patients AS (
    SELECT DISTINCT p.subject_id
    FROM `physionet-data.mimiciii_clinical.patients` p
    INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d
    ON p.subject_id = d.subject_id
    WHERE d.icd9_code IN ('4280', '4281', '42821', '42823', '42832', '40211', '40411', '40491')
),

-- Common heart failure medication classes
heart_failure_meds AS (
    SELECT drug_name_generic,
    CASE 
        WHEN LOWER(drug_name_generic) LIKE '%furosemide%' OR 
             LOWER(drug_name_generic) LIKE '%bumetanide%' OR 
             LOWER(drug_name_generic) LIKE '%torsemide%' OR
             LOWER(drug_name_generic) LIKE '%hydrochlorothiazide%' OR
             LOWER(drug_name_generic) LIKE '%chlorthalidone%' OR
             LOWER(drug_name_generic) LIKE '%metolazone%' THEN 'Diuretic'
        WHEN LOWER(drug_name_generic) LIKE '%lisinopril%' OR 
             LOWER(drug_name_generic) LIKE '%enalapril%' OR 
             LOWER(drug_name_generic) LIKE '%captopril%' OR
             LOWER(drug_name_generic) LIKE '%ramipril%' THEN 'ACE Inhibitor'
        WHEN LOWER(drug_name_generic) LIKE '%losartan%' OR 
             LOWER(drug_name_generic) LIKE '%valsartan%' OR 
             LOWER(drug_name_generic) LIKE '%candesartan%' OR
             LOWER(drug_name_generic) LIKE '%irbesartan%' THEN 'ARB'
        WHEN LOWER(drug_name_generic) LIKE '%metoprolol%' OR 
             LOWER(drug_name_generic) LIKE '%carvedilol%' OR 
             LOWER(drug_name_generic) LIKE '%bisoprolol%' OR
             LOWER(drug_name_generic) LIKE '%atenolol%' THEN 'Beta Blocker'
        WHEN LOWER(drug_name_generic) LIKE '%spironolactone%' OR 
             LOWER(drug_name_generic) LIKE '%eplerenone%' THEN 'Aldosterone Antagonist'
        WHEN LOWER(drug_name_generic) LIKE '%digoxin%' THEN 'Digoxin'
        WHEN LOWER(drug_name_generic) LIKE '%hydralazine%' THEN 'Hydralazine'
        WHEN LOWER(drug_name_generic) LIKE '%isosorbide%' THEN 'Nitrate'
        WHEN LOWER(drug_name_generic) LIKE '%sacubitril%' OR
             LOWER(drug_name_generic) LIKE '%entresto%' THEN 'ARNI'
        ELSE 'Other'
    END AS medication_class
    FROM `physionet-data.mimiciii_clinical.prescriptions`
)

SELECT 
    hfm.medication_class,
    COUNT(*) as prescription_count
FROM `physionet-data.mimiciii_clinical.prescriptions` pr
INNER JOIN heart_failure_patients hf ON pr.subject_id = hf.subject_id
INNER JOIN heart_failure_meds hfm ON pr.drug_name_generic = hfm.drug_name_generic
WHERE hfm.medication_class != 'Other'
GROUP BY hfm.medication_class
ORDER BY prescription_count DESC
LIMIT 10
```
</div>
</details>



<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">

- medication_class: Diuretic
  prescription_count: 8353710042
- medication_class: Beta Blocker
  prescription_count: 3754756003
- medication_class: ACE Inhibitor
  prescription_count: 244408010
- medication_class: Hydralazine
  prescription_count: 113171346
- medication_class: Digoxin
  prescription_count: 34102998
- medication_class: Nitrate
  prescription_count: 9392805
- medication_class: Aldosterone Antagonist
  prescription_count: 7319852
- medication_class: ARB
  prescription_count: 5011914
</div>
</details>

<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant</summary>
  <div markdown="1">
Let me get a more accurate count of the specific medications for heart failure patients:

Using tool: **query_db** with input:
```sql
WITH heart_failure_patients AS (
    SELECT DISTINCT p.subject_id
    FROM `physionet-data.mimiciii_clinical.patients` p
    INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d
    ON p.subject_id = d.subject_id
    WHERE d.icd9_code IN ('4280', '4281', '42821', '42823', '42832', '40211', '40411', '40491')
),

-- Common heart failure medications
heart_failure_meds AS (
    SELECT drug_name_generic
    FROM `physionet-data.mimiciii_clinical.prescriptions`
    WHERE 
        LOWER(drug_name_generic) LIKE '%furosemide%' OR 
        LOWER(drug_name_generic) LIKE '%bumetanide%' OR 
        LOWER(drug_name_generic) LIKE '%torsemide%' OR
        LOWER(drug_name_generic) LIKE '%hydrochlorothiazide%' OR
        LOWER(drug_name_generic) LIKE '%chlorthalidone%' OR
        LOWER(drug_name_generic) LIKE '%metolazone%' OR
        LOWER(drug_name_generic) LIKE '%lisinopril%' OR 
        LOWER(drug_name_generic) LIKE '%enalapril%' OR 
        LOWER(drug_name_generic) LIKE '%captopril%' OR
        LOWER(drug_name_generic) LIKE '%ramipril%' OR
        LOWER(drug_name_generic) LIKE '%losartan%' OR 
        LOWER(drug_name_generic) LIKE '%valsartan%' OR 
        LOWER(drug_name_generic) LIKE '%candesartan%' OR
        LOWER(drug_name_generic) LIKE '%irbesartan%' OR
        LOWER(drug_name_generic) LIKE '%metoprolol%' OR 
        LOWER(drug_name_generic) LIKE '%carvedilol%' OR 
        LOWER(drug_name_generic) LIKE '%bisoprolol%' OR
        LOWER(drug_name_generic) LIKE '%atenolol%' OR
        LOWER(drug_name_generic) LIKE '%spironolactone%' OR 
        LOWER(drug_name_generic) LIKE '%eplerenone%' OR
        LOWER(drug_name_generic) LIKE '%digoxin%' OR
        LOWER(drug_name_generic) LIKE '%hydralazine%' OR
        LOWER(drug_name_generic) LIKE '%isosorbide%'
    GROUP BY drug_name_generic
)

SELECT 
    pr.drug_name_generic,
    COUNT(*) as prescription_count
FROM `physionet-data.mimiciii_clinical.prescriptions` pr
INNER JOIN heart_failure_patients hf ON pr.subject_id = hf.subject_id
INNER JOIN heart_failure_meds hfm ON pr.drug_name_generic = hfm.drug_name_generic
GROUP BY pr.drug_name_generic
ORDER BY prescription_count DESC
LIMIT 10
```
</div>
</details>


<details style="width: 100%; position: relative; border: 1px solid #4287f5; border-radius: 8px; margin: 16px 0; background-color: #e6f0ff; color: #333; padding: 8px;">
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #1a56db;">Tool Result</summary>
  <div markdown="1">
- drug_name_generic: Furosemide
  prescription_count: 67710
- drug_name_generic: Metoprolol
  prescription_count: 31901
- drug_name_generic: Metoprolol Tartrate
  prescription_count: 22769
- drug_name_generic: Lisinopril
  prescription_count: 8858
- drug_name_generic: Captopril
  prescription_count: 6711
- drug_name_generic: HydrALAzine
  prescription_count: 5271
- drug_name_generic: Digoxin
  prescription_count: 4786
- drug_name_generic: Carvedilol
  prescription_count: 4009
- drug_name_generic: Hydralazine HCl
  prescription_count: 2931
- drug_name_generic: Isosorbide Mononitrate (Extended Release)
  prescription_count: 2019

</div>
</details>

<details style="width: 100%; position: relative; border: 1px solid #38b2ac; border-radius: 8px; margin: 16px 0; background-color: #e6fffa; color: #333; padding: 8px;" open>
  <summary style="cursor: pointer; padding: 8px; font-weight: bold; color: #047857;">Assistant Final Response</summary>
  <div markdown="1">
Based on the analysis of the MIMIC-III database, here are the most commonly prescribed medications for heart failure patients:

## Most Commonly Prescribed Medications for Heart Failure Patients

1. Furosemide (67,710 prescriptions) - A loop diuretic used to reduce fluid overload and congestion
2. Metoprolol (31,901 prescriptions) - A beta-blocker that helps reduce heart rate and blood pressure
3. Metoprolol Tartrate (22,769 prescriptions) - A specific formulation of metoprolol
4. Lisinopril (8,858 prescriptions) - An ACE inhibitor that helps relax blood vessels
5. Captopril (6,711 prescriptions) - Another ACE inhibitor
6. Hydralazine (5,271 prescriptions) - A vasodilator that relaxes blood vessels
7. Digoxin (4,786 prescriptions) - Increases the strength of heart contractions
8. Carvedilol (4,009 prescriptions) - A beta-blocker with additional vasodilating properties
9. Hydralazine HCl (2,931 prescriptions) - Another formulation of hydralazine
10. Isosorbide Mononitrate (2,019 prescriptions) - A nitrate that helps dilate blood vessels

## By Medication Class
The most common medication classes for heart failure patients are:

1. Diuretics - Used to reduce fluid retention and congestion
2. Beta Blockers - Help reduce heart rate and workload on the heart
3. ACE Inhibitors - Relax blood vessels and reduce blood pressure
4. Hydralazine - A vasodilator often used in combination with nitrates
5. Digoxin - Increases the force of heart contractions
6. Nitrates - Dilate blood vessels to reduce workload on the heart
7. Aldosterone Antagonists - Help manage fluid balance and protect the heart
8. ARBs (Angiotensin Receptor Blockers) - Similar to ACE inhibitors but work through a different mechanism

These findings align with standard heart failure treatment guidelines, which typically include diuretics for symptom relief and medications like ACE inhibitors, beta-blockers, and aldosterone antagonists to improve survival and reduce hospitalizations.
</div>
</details>


