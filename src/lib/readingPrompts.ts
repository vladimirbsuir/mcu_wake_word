import type { WordGroupData } from "@/lib/voice";

// General sentences to read aloud as "other speech": everyday requests and
// remarks in a normal voice. They are not tied to any wake phrase; words that
// sound like the phrase come from the LLM word lists of the Phrases step.
const BUILT_IN: Record<string, string[]> = {
  ru: [
    "Включи, пожалуйста, свет на кухне.",
    "Какая погода будет завтра утром?",
    "Поставь будильник на семь тридцать.",
    "Давай встретимся в четверг вечером.",
    "Во вторник я работаю из дома.",
    "Сделай музыку немного потише.",
    "Сколько времени займёт дорога до работы?",
    "Скорее всего, рейс задержат на час.",
    "Выключи телевизор и иди спать.",
    "Сколько стоит доставка до дома?",
    "Я слышу какой-то странный звук.",
    "Кот спрятался под диваном и не выходит.",
    "Она сдала экзамен на отлично.",
    "Сколько градусов сейчас на улице?",
    "Открой окно, здесь очень душно.",
    "На этой неделе у нас много встреч.",
    "Где-то рядом залаяла собака.",
    "Эта вода слишком холодная для купания.",
    "Перезвони мне, когда сможешь.",
    "Свет в коридоре опять мигает.",
    "Давай закажем пиццу на ужин.",
    "Слева от дома растёт старая берёза.",
    "Сосед сверлит стену с самого утра.",
    "Сегодня я весь день провёл дома.",
    "Скажи, пожалуйста, который час.",
    "Наш поезд отправляется через десять минут.",
    "Запиши меня к врачу на пятницу.",
    "Сначала сделай уроки, потом гуляй.",
    "Купи хлеба, молока и немного сыра.",
    "Расскажи мне что-нибудь интересное.",
    "Мы сидели в саду до самого вечера.",
    "Напомни мне позвонить маме вечером.",
    "Батарея телефона почти разрядилась.",
    "Дождь идёт с самого утра.",
    "Прочитай последние новости.",
    "Добавь яблоки в список покупок.",
    "Встреча перенесена на следующую неделю.",
    "Где-то здесь были мои ключи.",
    "Поставь чайник, пожалуйста.",
    "Завтра обещают сильный ветер.",
  ],
  en: [
    "Turn on the lights in the kitchen, please.",
    "What will the weather be like tomorrow morning?",
    "Set an alarm for seven thirty.",
    "Let's meet on Thursday evening.",
    "Could you turn the music down a little?",
    "How long does it take to get to work?",
    "The train leaves in ten minutes.",
    "I think I heard someone at the door.",
    "Remind me to call my sister later.",
    "We should order pizza for dinner.",
    "The meeting has been moved to Friday.",
    "Open the window, it's really stuffy in here.",
    "How much does delivery cost?",
    "Tell me something interesting.",
    "The cat is hiding under the sofa again.",
    "Please add milk and bread to the shopping list.",
    "My phone battery is almost empty.",
    "Switch off the TV and go to sleep.",
    "It has been raining since early morning.",
    "Read me the latest news headlines.",
  ],
};

const WORDS_PER_PROMPT = 3;

function normalize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

export function containsPhrase(text: string, phrase: string): boolean {
  const tokens = normalize(text);
  const target = normalize(phrase);
  if (!target.length) return false;
  for (let start = 0; start + target.length <= tokens.length; start += 1) {
    if (target.every((token, offset) => tokens[start + offset] === token)) return true;
  }
  return false;
}

function chunk(items: string[], size: number): string[] {
  const out: string[] = [];
  for (let index = 0; index < items.length; index += size) {
    out.push(items.slice(index, index + size).join(", "));
  }
  return out;
}

/** How many prompts come from the phrase-specific word lists. */
export function phraseSpecificCount(groups: Record<string, WordGroupData> | undefined): number {
  const words = [...(groups?.confusable?.items ?? []), ...(groups?.partial?.items ?? [])];
  return Math.ceil(words.length / WORDS_PER_PROMPT) + (groups?.everyday?.items?.length ?? 0);
}

/**
 * Prompts for "other speech" takes: lists of similar-sounding words and
 * everyday phrases from the Phrases step, interleaved with built-in sentences.
 * Anything that contains the whole wake phrase is left out.
 */
export function readingPrompts(
  phrase: string,
  language: string,
  groups: Record<string, WordGroupData> | undefined,
): string[] {
  const words = chunk(
    [...(groups?.confusable?.items ?? []), ...(groups?.partial?.items ?? [])],
    WORDS_PER_PROMPT,
  );
  const phrases = groups?.everyday?.items ?? [];
  const sentences = BUILT_IN[language] ?? [];
  const sources = [words, sentences, phrases];
  const longest = Math.max(0, ...sources.map((source) => source.length));
  const prompts: string[] = [];
  for (let index = 0; index < longest; index += 1) {
    for (const source of sources) {
      if (index < source.length) prompts.push(source[index]);
    }
  }
  return [...new Set(prompts)].filter((prompt) => !containsPhrase(prompt, phrase));
}
