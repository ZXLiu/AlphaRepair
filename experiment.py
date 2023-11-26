import os
import re
import subprocess
import time
import torch
import argparse

from transformers import RobertaTokenizer, RobertaForMaskedLM

from simple_template import generate_template, remove_redudant, generate_match_template, match_simple_operator
from tool.logger import Logger
from tool.fault_localization import get_location
from tool.d4j import build_d4j1_2
from validate_patches import GVpatches, UNIAPRpatches
from bert_beam_search import BeamSearch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def comment_remover(text):  # 用于删除注释
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # 注意:一个空格，而不是一个空字符串
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def add_new_line(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret_before = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    post_code = data[line_loc:]
    old_code = data[line_loc].strip()
    masked_line = " " + mask_token * 20 + " "
    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join(
            [x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + masked_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print("Context After:\n{}".format(post_code_input))

    # Straight up line replacement
    for token_len in range(1, 30):  # Within 10

        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret_before.append(("".join(beam[2]), beam[0] / token_len, "Before " + masked_line))
    ret_before.sort(key=lambda x: x[1], reverse=True)
    ret_before = remove_redudant(ret_before)  # 去除生成补丁中重复的补丁

    ret = []
    ret.extend(ret_before)
    ret.sort(key=lambda x: x[1], reverse=True)

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def process_file(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    fault_line = comment_remover(data[line_loc].strip())  # remove comments
    old_code = data[line_loc].strip()
    post_code = data[line_loc + 1:]

    line_size = 100

    while (1):
        pre_code_input = "</s> " + " ".join([x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + fault_line + post_code_input, return_tensors='pt')['input_ids'].size()[1] < 490:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))  # actual context len =  2*line_size
    print("Context Before:\n{}".format(pre_code_input))
    print(">> {} <<".format(fault_line))
    print("Context After:\n{}".format(post_code_input))

    fault_line_token_size = tokenizer(fault_line, return_tensors='pt')["input_ids"].shape[1] - 2

    # Straight up line replacement
    # 执行全部掩码(整个bug行全部用<mask>替换)
    for token_len in range(fault_line_token_size - 5, fault_line_token_size + 5):  # Within 10
        if token_len <= 0:
            continue
        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret.append(("".join(beam[2]), beam[0] / token_len, masked_line))

    # 执行部分掩码(在bug行生成只有前端有掩码或者只有后端有掩码的模板)
    templates = generate_template(fault_line)
    for partial_beginning, partial_end in templates:
        temp_size = fault_line_token_size - (
                tokenizer(partial_beginning, return_tensors='pt')["input_ids"].shape[1] - 2) - (
                            tokenizer(partial_end, return_tensors='pt')["input_ids"].shape[1] - 2)
        for token_len in range(2, 11):
            if token_len <= 0:
                continue
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len, masked_line))

    # 执行模板掩码(在bug行对于函数调用以及条件语句生成对于含有<mask>的模板)
    match_template = generate_match_template(fault_line, tokenizer)
    for match, length in match_template:
        for token_len in range(1, length + 5):
            if len(match.split(mask_token)) == 2:
                masked_line = " " + match.split(mask_token)[0] + mask_token * token_len + match.split(mask_token)[
                    1] + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    ret.append((match.split(mask_token)[0] + "".join(beam[2]) + match.split(mask_token)[1],
                                beam[0] / token_len, masked_line))
            else:
                masked_line = " "
                masked_line += (mask_token * token_len).join(match.split(mask_token)) + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    index = 0
                    gen_line = ""
                    for c in masked_line.split(mask_token)[:-1]:
                        gen_line += c
                        gen_line += beam[2][index]
                        index += 1
                    gen_line += masked_line.split(mask_token)[-1]
                    gen_line = gen_line[1:-1]
                    ret.append((gen_line, beam[0] / (token_len * (len(match.split(mask_token)) - 1)), masked_line))

    # 执行模板掩码(将操作符(如:!=)替换为<mask>)
    simple_operator_template = match_simple_operator(fault_line, tokenizer)
    for template in simple_operator_template:
        token_len = template.count("<mask>")
        masked_line = " " + template + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            index = 0
            gen_line = ""
            for c in masked_line.split(mask_token)[:-1]:
                gen_line += c
                gen_line += beam[2][index]
                index += 1
            gen_line += masked_line.split(mask_token)[-1]
            gen_line = gen_line[1:-1]
            ret.append((gen_line, beam[0] / token_len, masked_line))

    ret.sort(key=lambda x: x[1], reverse=True)
    ret = remove_redudant(ret)
    print("Generated Patches: " + str(len(ret)))

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def main(bug_ids, output_folder, skip_validation, uniapr, beam_width, re_rank, perfect, top_n_patches):
    if bug_ids[0] == 'none':
        bug_ids = build_d4j1_2()
    # model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    dirname = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dirname, 'CodeBERT')
    model = RobertaForMaskedLM.from_pretrained(path).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(path)

    for bug_id in bug_ids:
        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        subprocess.run(
            ["/home/lzx/defects4j/framework/bin/defects4j", "checkout", "-p", bug_id.split('-')[0], "-v", bug_id.split('-')[1] + 'b', "-w", '/tmp/' + bug_id],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
        #     bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('/tmp/' + bug_id)), shell=True)
        patch_pool_folder = "patches-pool"
        
        location = get_location(bug_id, perfect=perfect)
        print(location[0][0])
        if perfect:
            patch_pool_folder = "pfl-patches-pool-temp"

        # testmethods = os.popen('defects4j export -w %s -p tests.trigger' % ('/tmp/' + bug_id)).readlines()
        result = subprocess.run(["/home/lzx/defects4j/framework/bin/defects4j", "export", "-w", "/tmp/"+bug_id, "-p", "tests.trigger"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 判断输出结果是否为空
        if result.stdout:
            # 获取输出结果的最后一行，并去除空格和换行符
            testmethods = result.stdout.decode().splitlines()
        else:
            # 输出错误信息
            print(result.stderr.decode())

        path = "/home/lzx/PycharmProject/AlphaRepair/"+output_folder + '/' + bug_id + "_result.txt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger = Logger(output_folder + '/' + bug_id + "_result.txt")
        logger.logo(args)

        if uniapr:
            validator = UNIAPRpatches(bug_id, testmethods, logger, patch_pool_folder=patch_pool_folder, skip_validation=skip_validation)
        else:
            validator = GVpatches(bug_id, testmethods, logger, patch_pool_folder=patch_pool_folder, skip_validation=skip_validation)

        for file, line_number in location:
            print('Location: {} line # {}'.format(file, line_number))
            file = '/tmp/' + bug_id + '/' + file

            start_time = time.time()
            if len(location) > 3 and perfect:  # too many lines, can't really handle in time
                pre_code, fault_line, changes, post_code = process_file(file, line_number, tokenizer, model, 15,
                                                                        re_rank, top_n_patches)
            else:
                pre_code, fault_line, changes, post_code = process_file(file, line_number, tokenizer, model, beam_width,
                                                                        re_rank, top_n_patches)
            end_time = time.time()

            validator.add_new_patch_generation(pre_code, fault_line, changes, post_code, file, line_number,
                                               end_time - start_time)
        validator.validate()

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_id', type=str, default='Chart-12')
    parser.add_argument('--uniapr', action='store_true', default=False)
    parser.add_argument('--output_folder', type=str, default='codebert_result')
    parser.add_argument('--skip_v', action='store_true', default=False)
    parser.add_argument('--re_rank', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=25)
    parser.add_argument('--perfect', action='store_true', default=True)
    parser.add_argument('--top_n_patches', type=int, default=-1)
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    main([args.bug_id], args.output_folder, args.skip_v, args.uniapr, args.beam_width,
         args.re_rank, args.perfect, args.top_n_patches)
