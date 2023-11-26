import os
import subprocess


# 通过bug_id和perfect获取对应的文件路径
def get_loc_file(bug_id, perfect):  # bug_id:缺陷ID,perfect:是否使用完美缺陷定位
    # dirname = os.path.dirname(__file__)  # 获取当前脚本所在的目录的绝对路径
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.dirname(dirname)
    if perfect:
        loc_file = 'location/groundtruth/%s/%s' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    else:
        loc_file = 'location/ochiai/%s/%s.txt' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    loc_file = os.path.join(dirname, loc_file)  # 由bug_id获取到对应的文件路径
    if os.path.isfile(loc_file):  # 判断指定路径是否为一个文件
        return loc_file
    else:
        print(loc_file)
        return ""


# 从给定的bug_id获取位置信息
# perfect fault localization returns 1 line, top n gets top n lines for non-perfect FL (40 = decoder top n)
def get_location(bug_id, perfect=True, top_n=40):

    # source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + bug_id).readlines()[-1].strip() + "/"  # os.popen()用于执行一个shell命令
    project_name = bug_id.split('-')[0]
    id = bug_id.split('-')[1]
    # 将defects4j中对应的缺陷项目放到工作目录tmp中
    subprocess.run(['/home/lzx/defects4j/framework/bin/defects4j', 'checkout', '-p', project_name, '-v', id+'b', '-w', '/tmp/'+bug_id])
    # 获取指定bug_id的项目的源代码目录的绝对路径
    result = subprocess.run(["/home/lzx/defects4j/framework/bin/defects4j", "export", "-p", "dir.src.classes", "-w", "/tmp/" + bug_id],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 判断输出结果是否为空
    if result.stdout:
        # 获取输出结果的最后一行，并去除空格和换行符
        source_dir = result.stdout.decode().splitlines()[-1].strip() + "/"
    else:
        # 输出错误信息
        print(result.stderr.decode())
    location = []
    location_dict = {}
    loc_file = get_loc_file(bug_id, perfect)
    if loc_file == "":
        return location
    if perfect:  # 若是完美缺陷定位
        lines = open(loc_file, 'r').readlines()
        for loc_line in lines:
            loc_line = loc_line.split("||")[0]  # take first line in lump
            classname, line_id = loc_line.split(':')
            classname = ".".join(classname.split(".")[:-1])  # remove function name
            if '$' in classname:
                classname = classname[:classname.index('$')]
            file = source_dir + "/".join(classname.split(".")) + ".java"
            location.append((file, int(line_id) - 1))
    else:
        lines = open(loc_file, 'r').readlines()
        for loc_line in lines:
            loc_line = loc_line.split(",")[0]
            classname, line_id = loc_line.split("#")
            if '$' in classname:
                classname = classname[:classname.index('$')]
            file = source_dir + "/".join(classname.split(".")) + ".java"
            if file + line_id not in location_dict:
                location.append((file, int(line_id) - 1))
                location_dict[file + line_id] = 0
            else:
                print("Same Fault Location: {}, {}".format(file, line_id))
        pass

    return location[:top_n]


# if __name__ == '__main__':
#     loc_file = get_loc_file('Chart-1',True)
#     location = get_location('Chart-1',True)