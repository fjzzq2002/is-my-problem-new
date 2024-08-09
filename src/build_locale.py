# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8
from openai import AsyncOpenAI
from together import AsyncTogether
import anthropic
import hashlib
import asyncio
from tqdm.auto import tqdm

# from tqdm import tqdm
import time

start_time = time.time()


with open("settings.json") as f:
    settings = json.load(f)


from lingua import Language, LanguageDetectorBuilder
detector = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.9).build()
# ty chatgpt
lang_map = {
    Language.AFRIKAANS: ("za", "Afrikaans"),
    Language.ALBANIAN: ("al", "Albanian"),
    Language.ARABIC: ("sa", "Arabic"),
    Language.ARMENIAN: ("am", "Armenian"),
    Language.AZERBAIJANI: ("az", "Azerbaijani"),
    Language.BASQUE: ("eu", "Basque"),
    Language.BELARUSIAN: ("by", "Belarusian"),
    Language.BENGALI: ("bd", "Bengali"),
    Language.BOKMAL: ("no", "Bokmål"),
    Language.BOSNIAN: ("ba", "Bosnian"),
    Language.BULGARIAN: ("bg", "Bulgarian"),
    Language.CATALAN: ("ad", "Catalan"),
    Language.CHINESE: ("cn", "Chinese"),
    Language.CROATIAN: ("hr", "Croatian"),
    Language.CZECH: ("cz", "Czech"),
    Language.DANISH: ("dk", "Danish"),
    Language.DUTCH: ("nl", "Dutch"),
    Language.ENGLISH: ("gb", "English"),
    Language.ESPERANTO: ("eo", "Esperanto"),
    Language.ESTONIAN: ("ee", "Estonian"),
    Language.FINNISH: ("fi", "Finnish"),
    Language.FRENCH: ("fr", "French"),
    Language.GANDA: ("ug", "Ganda"),
    Language.GEORGIAN: ("ge", "Georgian"),
    Language.GERMAN: ("de", "German"),
    Language.GREEK: ("gr", "Greek"),
    Language.GUJARATI: ("in", "Gujarati"),
    Language.HEBREW: ("il", "Hebrew"),
    Language.HINDI: ("in", "Hindi"),
    Language.HUNGARIAN: ("hu", "Hungarian"),
    Language.ICELANDIC: ("is", "Icelandic"),
    Language.INDONESIAN: ("id", "Indonesian"),
    Language.IRISH: ("ie", "Irish"),
    Language.ITALIAN: ("it", "Italian"),
    Language.JAPANESE: ("jp", "Japanese"),
    Language.KAZAKH: ("kz", "Kazakh"),
    Language.KOREAN: ("kr", "Korean"),
    Language.LATIN: ("va", "Latin"),
    Language.LATVIAN: ("lv", "Latvian"),
    Language.LITHUANIAN: ("lt", "Lithuanian"),
    Language.MACEDONIAN: ("mk", "Macedonian"),
    Language.MALAY: ("my", "Malay"),
    Language.MAORI: ("nz", "Maori"),
    Language.MARATHI: ("in", "Marathi"),
    Language.MONGOLIAN: ("mn", "Mongolian"),
    Language.NYNORSK: ("no", "Nynorsk"),
    Language.PERSIAN: ("ir", "Persian"),
    Language.POLISH: ("pl", "Polish"),
    Language.PORTUGUESE: ("pt", "Portuguese"),
    Language.PUNJABI: ("in", "Punjabi"),
    Language.ROMANIAN: ("ro", "Romanian"),
    Language.RUSSIAN: ("ru", "Russian"),
    Language.SERBIAN: ("rs", "Serbian"),
    Language.SHONA: ("zw", "Shona"),
    Language.SLOVAK: ("sk", "Slovak"),
    Language.SLOVENE: ("si", "Slovene"),
    Language.SOMALI: ("so", "Somali"),
    Language.SOTHO: ("za", "Sotho"),
    Language.SPANISH: ("es", "Spanish"),
    Language.SWAHILI: ("ke", "Swahili"),
    Language.SWEDISH: ("se", "Swedish"),
    Language.TAGALOG: ("ph", "Tagalog"),
    Language.TAMIL: ("in", "Tamil"),
    Language.TELUGU: ("in", "Telugu"),
    Language.THAI: ("th", "Thai"),
    Language.TSONGA: ("za", "Tsonga"),
    Language.TSWANA: ("bw", "Tswana"),
    Language.TURKISH: ("tr", "Turkish"),
    Language.UKRAINIAN: ("ua", "Ukrainian"),
    Language.URDU: ("pk", "Urdu"),
    Language.VIETNAMESE: ("vn", "Vietnamese"),
    Language.WELSH: ("gb", "Welsh"),
    Language.XHOSA: ("za", "Xhosa"),
    Language.YORUBA: ("ng", "Yoruba"),
    Language.ZULU: ("za", "Zulu"),
}
def process_all_problems():
    fns = list(problem_filenames())
    for problem_file_cur in tqdm(fns):
        try:
            p = read_problem(problem_file_cur)
        except Exception as e:
            print('error',problem_file_cur,e)
            continue
        if 'locale' in p:
            continue
        detected = detector.detect_language_of(p["title"]+'\n'+p['statement'])
        rst = None
        if detected is None:
            rst = ('un', 'Unknown')
        else:
            rst = lang_map[detected]
        p['locale'] = rst
        dump_json_safe_utf8(p, problem_file_cur)

if __name__ == "__main__":
    process_all_problems()