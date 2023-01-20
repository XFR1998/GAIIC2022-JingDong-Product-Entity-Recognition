import re
import jieba
import jieba.posseg as psg  # 结巴分词的词性标注
from cnradical import Radical, RunOption


def is_chinese(char: str) -> bool:
    """
    判断字符是不是中文字符
    Args:
        char: 字符
    Returns:
        True: 是中文
        False: 不是中文
    """
    if "\u4e00" <= char <= "\u9fff":
        return True
    else:
        return False




def extract_word_flags_bounds(s, max_len):
    """
        提取字的词性和词位
    """
    # s:string
    # [CLS], [SEP], [PAD]token另算， 传进来的s不包含这些
    # 词性标x (非语素字只是一个符号，字母 x通常用于代表未知数、符号)
    # 词位标s（single）
    # 1. 提取词性和词位特征
    # 当前句子的词性和词位特征
    word_flags = []
    word_bounds = ["M"] * len(list(s))
    for word, flag in psg.cut(s):
        if len(word) == 1:  # 单独成词
            st_idx = len(word_flags)  # word_flags的长度，也就是我们已经处理了sentence中多少个字
            word_bounds[st_idx] = "S"
            word_flags.append(flag)
        else:
            pat = r"^([0-9A-Za-z]+)$"
            if re.match(pat, word):  # 如果是英文和数字的词，可能在原句中占了多个字的位置
                # if word.islower() or word.isupper():  # 如果是英文和数字的词，可能在原句中占了多个字的位置
                st_idx = len(word_flags)
                word_bounds[st_idx] = "B"  # 词首
                # 找出当前英文数字的词在字级别的原句list中，占了多少个位置
                add_idx = 1
                while True:
                    if "".join(s[st_idx: st_idx + add_idx]) == word:
                        break
                    else:
                        add_idx += 1
                word_flags += [flag] * add_idx
                ed_idx = len(word_flags) - 1
                word_bounds[ed_idx] = "E"  # 词尾
            else:
                st_idx = len(word_flags)
                word_bounds[st_idx] = "B"  # 词首
                word_flags += [flag] * len(word)
                ed_idx = len(word_flags) - 1
                word_bounds[ed_idx] = "E"  # 词尾

    word_flags = word_flags[:max_len - 2]
    word_bounds = word_bounds[:max_len - 2]
    # 这几个特殊token [CLS], [SEP], [PAD]也算上
    return_word_flags = []
    return_word_flags.extend('<CLS>')
    return_word_flags.extend(word_flags)
    return_word_flags.extend('<SEP>')
    return_word_flags.extend(['<PAD>']*(max_len-len(return_word_flags)))

    return_word_bounds = []
    return_word_bounds.extend('<CLS>')
    return_word_bounds.extend(word_bounds)
    return_word_bounds.extend('<SEP>')
    return_word_bounds.extend(['<PAD>']*(max_len-len(return_word_bounds)))
    assert len(return_word_flags) == len(return_word_bounds)

    return return_word_flags,return_word_bounds


def extract_pinyin(s, max_len):
    """
        提取字的拼音
     """
    # s: string
    # [CLS], [SEP], [PAD]token另算， 传进来的s不包含这些

    s = list(s)
    pinyin = Radical(RunOption.Pinyin)  # 获取拼音
    pinyin_list = [pinyin.trans_ch(word) if is_chinese(word) and pinyin.trans_ch(word) else '<SPECIAL>' for word in s]
    pinyin_list = pinyin_list[:max_len-2]

    return_pinyin_list = []
    return_pinyin_list.extend(['<CLS>'])
    return_pinyin_list.extend(pinyin_list)
    return_pinyin_list.extend(['<SEP>'])
    return_pinyin_list.extend(['<PAD>'] * (max_len - len(return_pinyin_list)))

    return return_pinyin_list

def extract_radical(s, max_len):
    """
        提取字的偏旁
    """
    # s: string
    # [CLS], [SEP], [PAD]token另算， 传进来的s不包含这些

    s = list(s)
    radical = Radical(RunOption.Radical)  # 获取偏旁
    radical_list = [radical.trans_ch(word) if is_chinese(word) and radical.trans_ch(word) else '<SPECIAL>' for word in s]
    radical_list = radical_list[:max_len - 2]

    return_radical_list = []
    return_radical_list.extend(['<CLS>'])
    return_radical_list.extend(radical_list)
    return_radical_list.extend(['<SEP>'])
    return_radical_list.extend(['<PAD>'] * (max_len - len(return_radical_list)))



    return return_radical_list