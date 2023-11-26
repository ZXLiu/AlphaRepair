import os
import re
import signal
import subprocess
import time
import javalang
import shutil


# Basic generate and valdiate class
class GVpatches(object):

    def __init__(self, bug_id, testmethods, logger, patch_pool_folder="patches-pool", skip_validation=False):
        self.bug_id = bug_id
        self.pre_codes = []
        self.fault_lines = []
        self.l_changes = []
        self.post_codes = []
        self.files = []
        self.line_numbers = []
        self.num_of_patches = 1
        self.generation_time = 0
        self.testmethods = testmethods
        self.logger = logger
        self.patch_pool_folder = patch_pool_folder
        self.skip_validation = skip_validation
        # self.src_dir = src_dir

    # 添加一个新补丁到validator中
    def add_new_patch_generation(self, pre_code, fault_line, changes, post_code, file, line_number, n_time):
        self.pre_codes.append(pre_code)
        self.fault_lines.append(fault_line)
        self.l_changes.append(changes)
        self.post_codes.append(post_code)
        self.files.append(file)
        self.line_numbers.append(line_number)
        self.generation_time += n_time

    # 将对应的缺陷项目导出到工作目录tmp中
    def checkout_d4j_project(self):
        subprocess.run('rm -rf ' + '/tmp/' + self.bug_id, shell=True)
        subprocess.run(
            ["/home/lzx/defects4j/framework/bin/defects4j", "checkout", "-p", self.bug_id.split('-')[0], "-v",
             self.bug_id.split('-')[1] + 'b', "-w", '/tmp/' + self.bug_id],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # subprocess.run("defects4j checkout -p %s -v %s -w %s" % (self.bug_id.split('-')[0],self.bug_id.split('-')[1] + 'b',('/tmp/' + self.bug_id)),
        #                shell=True)
        # shutil.copytree(self.src_dir, '/tmp/' + self.bug_id)
        # subprocess.run('mv ' + '/tmp/' + self.bug_id + '_buggy' + ' /tmp/'+self.bug_id, shell=True)

    # 将生成的补丁写入有错误的文件中进行重新编译
    def write_changes_to_file(self, change, index):
        with open(self.files[index], 'w', encoding='utf-8') as f:
            for line in self.pre_codes[index]:
                f.write(line)
            f.write(change + '\n')
            print(change)
            for line in self.post_codes[index]:
                f.write(line)
        subprocess.run("touch -r " + self.files[index] + " -d '1 day' " + self.files[index],
                       shell=True)  # 这将强制重新编译

    # 判断生成的补丁是否合理(即:能否通过所有的测试案例)
    def run_d4j_test(self, prob, index):

        bugg = False
        compile_fail = False
        timed_out = False
        entire_bugg = False
        error_string = ""

        with open(self.files[index], 'r', encoding='utf-8') as f:
            tmpcode = f.read()
            try:
                # 将Java代码转换为抽象语法树（Abstract Syntax Tree，AST），以便进一步分析和处理代码
                tokens = javalang.tokenizer.tokenize(tmpcode)
                parser = javalang.parser.Parser(tokens)
                parser.parse()
            except:
                self.logger.logo(prob)
                self.logger.logo("Syntax Error")
                return
        # 对补丁执行之前触发该bug的测试用例
        for t in self.testmethods:
            # cmd = 'defects4j test -w %s/ -t %s' % (('/tmp/' + self.bug_id), t.strip())
            Returncode = ""
            error_file = open("stderr.txt", "wb")
            child = subprocess.Popen(["/home/lzx/defects4j/framework/bin/defects4j", "test", "-w", '/tmp/' + self.bug_id + '/', "-t", t.strip()],
                                     stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                     start_new_session=True)
            while_begin = time.time()
            while True:
                Flag = child.poll()  # 检查子进程是否已经完成
                if Flag == 0:  # 子进程执行完成
                    Returncode = child.stdout.readlines()  # child.stdout.read()
                    print(b"".join(Returncode).decode('utf-8'))
                    error_file.close()
                    break
                elif Flag != 0 and Flag is not None:
                    compile_fail = True
                    error_file.close()
                    with open("stderr.txt", "rb") as f:
                        r = f.readlines()
                    for line in r:
                        if re.search(':\serror:\s', line.decode('utf-8')):
                            error_string = line.decode('utf-8')
                            break
                    print(error_string)
                    break
                elif time.time() - while_begin > 15:
                    error_file.close()
                    print('ppp')
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    timed_out = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":  # 说明当前测试用例执行成功，继续执行下一个测试用例
                continue
            else:  # 当前测试用例执行失败(即:该补丁未能修复bug)
                print("Failure Output")
                bugg = True
                break

        # 然后我们检查它是否通过了所有的测试，包括之前的ok测试
        if not bugg:
            print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
            # cmd = 'defects4j test -w %s/' % ('/tmp/' + self.bug_id)
            Returncode = ""
            child = subprocess.Popen(["/home/lzx/defects4j/framework/bin/defects4j", "test", "-w", '/tmp/' + self.bug_id + '/'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                     start_new_session=True)
            while_begin = time.time()
            while True:
                Flag = child.poll()  # 检查子进程是否已经完成
                # print(time.time() - while_begin, Flag)
                if Flag == 0:  # 通过所有测试用例
                    Returncode = child.stdout.readlines()  # child.stdout.read()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.time() - while_begin > 180:
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            print(log)
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                print('success')
                endtime = time.time()
            else:
                entire_bugg = True

        self.logger.logo(prob)
        if compile_fail:
            self.logger.logo("Compiled Error:")
            self.logger.logo(error_string)
            self.logger.logo("Compilation Failed")
        elif timed_out:
            self.logger.logo("Timed Out")
        elif bugg:
            self.logger.logo("Failed Testcase")
        elif entire_bugg:
            self.logger.logo("Failed Original Testcase")
        else:
            self.logger.logo("Success (Plausible Patch)")
            return True
        
        return False

    def validate(self):
        start_time = time.time()
        print("----------------- BEGIN VALIDATION OF PATCHES -----------------")
        for index in range(len(self.l_changes)):
            self.logger.logo("Validating Patches for file: {} {}".format(self.files[index], self.line_numbers[index]))
            self.checkout_d4j_project()
            for change, prob, masked_line in self.l_changes[index]:
                self.write_changes_to_file(change, index)
                self.logger.logo("----------------------------------------")
                self.logger.logo("Patch Number :{}".format(self.num_of_patches))
                self.logger.logo('- ' + self.fault_lines[index].strip())
                self.logger.logo('+ ' + change.strip())
                self.logger.logo('mask:' + masked_line.strip())

                if not self.skip_validation:
                    is_plausible = self.run_d4j_test(prob, index)
                    if is_plausible:
                        diff_path = os.path.join(self.patch_pool_folder, "{}.diff".format(self.num_of_patches))
                        with open(diff_path, "w") as f:
                            f.write('- ' + self.fault_lines[index].strip() + "\n")
                            f.write('+ ' + change.strip())
                    
                self.num_of_patches += 1
        end_time = time.time()
        self.logger.logo("----------------- FINISH VALIDATION OF PATCHES -----------------")
        # Todo: Maybe change log to separate log file
        self.logger.logo("Patch Generation Time: {}".format(self.generation_time))
        self.logger.logo("Patch Validation Time: {}".format(end_time - start_time))


class UNIAPRpatches(GVpatches):
    D4J_Struct = {
        'Chart': {
            'source': 'org',
            'classpath': 'build/'
        },
        'Closure': {
            'source': 'com',
            'classpath': 'build/classes:lib/*:build/lib/*'
        },
        'Lang': {
            'source': 'org',
            'classpath': 'target/classes'
        },
        'Math': {
            'source': 'org',
            'classpath': 'target/classes'
        },
        'Mockito': {
            'source': 'org',
            'classpath': 'target/classes'
        },
        'Time': {
            'source': 'org',
            'classpath': 'target/classes'
        }
    }

    def javac_compile(self, index):
        compiled = True
        try:
            subprocess.run("cd /tmp/" + self.bug_id + "; javac -cp " + self.D4J_Struct[self.bug_id.split("-")[0]][
                'classpath'] + " " + self.files[index].split(self.bug_id + "/")[-1],
                           shell=True, check=True, timeout=10)
        except subprocess.CalledProcessError as e:
            self.logger.logo(e.stderr)
            compiled = False
        except subprocess.TimeoutExpired as e:
            self.logger.logo(e.stderr)
            compiled = False
        return compiled

    def compile_d4j_project(self):
        compiled = True
        try:
            subprocess.run(["/home/lzx/defects4j/framework/bin/defects4j", "compile", "-w", '/tmp/' + self.bug_id],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            # print("Return code: {}".format(e.returncode))
            compiled = False

        return compiled

    def make_patch_pool_folder(self):
        maven_project_path = "~/prapr_data/Projects/" + self.bug_id.split("-")[0] + "/" + self.bug_id.split("-")[
            1] + "/" + self.patch_pool_folder
        subprocess.run('rm -rf ' + maven_project_path, shell=True)
        subprocess.run('mkdir ' + maven_project_path, shell=True)
        return

    def move_compiled_patch(self, index):
        # assert (len(class_changed) > 0)
        class_file = self.files[index].replace(".java", ".class")
        maven_project_path = "~/prapr_data/Projects/" + self.bug_id.split("-")[0] + "/" + self.bug_id.split("-")[
            1] + "/" + self.patch_pool_folder
        source_folder = self.D4J_Struct[self.bug_id.split("-")[0]]['source']
        partial_path = ""
        split_file = class_file.split("/")
        for i, file in enumerate(split_file):
            if file == source_folder:
                partial_path = "/".join(split_file[i:])
                break

        folder_name = "/".join(partial_path.split("/")[:-1])
        subprocess.run('mkdir -p ' + maven_project_path + "/" + str(self.num_of_patches) + "/" + folder_name,
                       shell=True)
        subprocess.run(
            'mv ' + class_file + " " + maven_project_path + "/" + str(self.num_of_patches) + "/" + partial_path,
            shell=True)
        subprocess.run('cp ' + self.files[index] + " " + maven_project_path + "/" + str(self.num_of_patches) + "/",
                       shell=True)

        return

    def syntax_check(self, index):
        with open(self.files[index], 'r', encoding='utf-8') as f:
            tmpcode = f.read()
            try:
                tokens = javalang.tokenizer.tokenize(tmpcode)
                parser = javalang.parser.Parser(tokens)
                parser.parse()
            except:
                return False
        return True

    def run_uniapr(self):
        success = True
        try:
            # grab basic uniapr output
            uniapr_val_file = "~/CodeBertAPR/" + self.logger.logfile.split(".txt")[0] + "_restart_val.txt"
            subprocess.run("cd ~/prapr_data/Projects/" + self.bug_id.split("-")[0] + "/" + self.bug_id.split("-")[1] +
                           "; mvn org.uniapr:uniapr-plugin:validate -DrestartJVM=true -Dd4jAllTestsFile=all_tests "
                           "-DpatchesPool=" + self.patch_pool_folder + ">" + uniapr_val_file,
                           shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            success = False
        return success

    def validate(self):  # different validation process than traditional
        # make folder
        num_of_compiled_patches = 0
        file_patches = {}
        start_time = time.time()
        self.make_patch_pool_folder()
        self.D4J_Struct[self.bug_id.split("-")[0]]['classpath'] = \
            subprocess.Popen(
                ["/home/lzx/defects4j/framework/bin/defects4j", "export", "-w", '/tmp/' + self.bug_id, "-p",
                 "cp.compile"],stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().splitlines()[-1]
        self.D4J_Struct[self.bug_id.split("-")[0]]['classpath'] += ":" + subprocess.Popen(
                ["/home/lzx/defects4j/framework/bin/defects4j", "export", "-w", '/tmp/' + self.bug_id, "-p",
                 "cp.test"],stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().splitlines()[-1]
        # self.D4J_Struct[self.bug_id.split("-")[0]]['classpath'] = \
            # os.popen('defects4j export -w %s -p cp.compile' % ('/tmp/' + self.bug_id)).readlines()[-1]
        # self.D4J_Struct[self.bug_id.split("-")[0]]['classpath'] += ":" + os.popen('defects4j export -w %s -p cp.test' % ('/tmp/' + self.bug_id)).readlines()[-1]
        self.logger.logo("Class path: {}".format(self.D4J_Struct[self.bug_id.split("-")[0]]['classpath']))
        print("----------------- BEGIN VALIDATION OF PATCHES -----------------")
        for index in range(len(self.l_changes)):
            file_patches[self.files[index] + " " + str(self.line_numbers[index])] = \
                {'total': 0, 'compiled': 0, 'syntax error': 0}
            self.logger.logo(
                "---- Validating Patches for file: {} {} ----".format(self.files[index], self.line_numbers[index]))
            self.checkout_d4j_project()
            self.compile_d4j_project()
            for change, prob, masked_line in self.l_changes[index]:
                self.write_changes_to_file(change, index)
                self.logger.logo("----------------------------------------")
                self.logger.logo("Patch Number :{}".format(self.num_of_patches))
                self.logger.logo('- ' + self.fault_lines[index].strip())
                self.logger.logo('+ ' + change.strip())
                self.logger.logo('mask:' + masked_line.strip())
                if not self.skip_validation:
                    syntax = self.syntax_check(index)
                    if syntax:
                        compiled = self.javac_compile(index)
                        if compiled:
                            num_of_compiled_patches += 1
                            self.move_compiled_patch(index)
                            file_patches[self.files[index] + " " + str(self.line_numbers[index])]['compiled'] += 1
                        else:
                            self.logger.logo("Compiled Error")
                    else:
                        self.logger.logo("Syntax Error")
                        file_patches[self.files[index] + " " + str(self.line_numbers[index])]['syntax error'] += 1
                self.num_of_patches += 1
                file_patches[self.files[index] + " " + str(self.line_numbers[index])]['total'] += 1
        compile_end_time = time.time()
        self.logger.logo("----------------- RUNNING UNIAPR VALIDATION -----------------")
        if not self.skip_validation:
           success = self.run_uniapr()
           if success:
               self.logger.logo("Uniapr Success!")
           else:
               self.logger.logo("Uniapr Failure :(")
        uniapr_end_time = time.time()
        self.logger.logo("----------------- FINISH VALIDATION OF PATCHES -----------------")
        self.logger.logo(
            "Num of patches generated: {} compiled: {}".format(self.num_of_patches, num_of_compiled_patches))
        for file_line, item in file_patches.items():
            self.logger.logo("{}: Total: {} Compiled: {} Syntax Errors: {}".format(file_line, item['total'],
                                                                                   item['compiled'],
                                                                                   item['syntax error']))
        self.logger.logo("Patch Generation Time: {}".format(self.generation_time))
        self.logger.logo("Patch Validation Time: {}".format(uniapr_end_time - start_time))
        self.logger.logo("Patch Compilation Time: {}".format(compile_end_time - start_time))
        self.logger.logo("Uniapr Time: {}".format(uniapr_end_time - compile_end_time))
