from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

llm = ChatOpenAI()
llm = ChatOpenAI(openai_api_key="sk-GcdtvRQev9LTDR8w915173693bFf40B897040f94D5582657", openai_api_base="https://api.xty.app/v1")
outputParse = StrOutputParser()

schema = Object(
    id="script",
    description="Adapted from the novel into script",
    attributes=[
        Text(
            id="role",
            description="The character who is speaking or performing an action",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        )
    ],
    examples=[
        (
            '''
         那张角本是个不第秀才，因入山采药，遇一老人，碧眼童颜，手执藜杖，唤角至一洞中，以天书三卷授之，曰：“此名《太平要术》，汝得之，当代天宣化，普救世人；若萌异心，必获恶报。”角拜问姓名。老人曰：“吾乃南华老仙也。”
            ''',
            [
                {"role": "南华老仙", "dialogue": "此名《太平要术》，汝得之，当代天宣化，普救世人；若萌异心，必获恶报。"},
                {"role": "南华老仙", "dialogue": "吾乃南华老仙也。"},
            ],
        ),
        (
            '''
           玄德幼时，与乡中小儿戏于树下，曰：“我为天子，当乘此车盖。”叔父刘元起奇其言，曰：“此儿非常人也！”
            ''',
            [
                {"role": "玄德", "dialogue": "我为天子，当乘此车盖。"},
                {"role": "刘元起", "dialogue": "此儿非常人也！"},
            ],
        )
    ],
    many=True,
)
with open("./data/test.txt",'r' ,encoding='utf-8') as file:
    content = file.read()


print(content)
chain = create_extraction_chain(llm, schema)
res = chain.invoke(content)
print(res)
for row in res['data']['script']:
    print(row)