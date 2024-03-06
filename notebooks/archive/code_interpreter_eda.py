from codeinterpreterapi import CodeInterpreterSession

async def main():
    # create a session
    session = CodeInterpreterSession()
    await session.astart()

    # generate a response based on user input
    response = await session.generate_response(
        "Plot the bitcoin chart of 2023 YTD"
    )

    # output the response (text + image)
    print("AI: ", response.content)
    for file in response.files:
        file.show_image()

    # terminate the session
    await session.astop()


if __name__ == "__main__":
    import asyncio
    # run the async function
    asyncio.run(main())