from test_net import run
import os
import argparse

import xml.dom
import xml.dom.minidom
from PIL import Image
import shutil
import sys


# create a simple element
def create_element_node(doc, tag, attr):

    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)
    element_node.appendChild(text_node)

    return element_node


# add a child node
def create_child_node(doc, tag, attr, parent_node):
    child_node = create_element_node(doc, tag, attr)
    parent_node.appendChild(child_node)


# create pascal voc object node
def create_object_node(doc, attrs, box):
    object_node = doc.createElement('object')
    create_child_node(doc, 'name', box[0], object_node)
    create_child_node(doc, 'pose', attrs['pose'], object_node)
    create_child_node(doc, 'truncated', attrs['truncated'], object_node)
    create_child_node(doc, 'difficult', attrs['difficult'], object_node)

    bndbox_node = doc.createElement('bndbox')
    create_child_node(doc, 'xmin', box[1], bndbox_node)
    create_child_node(doc, 'ymin', box[2], bndbox_node)
    create_child_node(doc, 'xmax', box[3], bndbox_node)
    create_child_node(doc, 'ymax', box[4], bndbox_node)
    object_node.appendChild(bndbox_node)

    return object_node


# create xml file
def create_xml_file(anno_file, attrs):

    my_dom = xml.dom.getDOMImplementation()
    doc = my_dom.createDocument(None, attrs['root'], None)

    root_node = doc.documentElement

    create_child_node(doc, 'folder', attrs['folder'], root_node)

    create_child_node(doc, 'filename', attrs['image_name'], root_node)

    source_node = doc.createElement('source')
    create_child_node(doc, 'database', attrs['database'], source_node)
    create_child_node(doc, 'annotation', attrs['annotation'], source_node)
    create_child_node(doc, 'image', 'flickr', source_node)
    create_child_node(doc, 'flickrid', 'NULL', source_node)
    root_node.appendChild(source_node)

    owner_node = doc.createElement('owner')
    create_child_node(doc, 'flickrid', 'NULL', owner_node)
    create_child_node(doc, 'name', attrs['author'], owner_node)
    root_node.appendChild(owner_node)

    size_node = doc.createElement('size')
    create_child_node(doc, 'width', attrs['width'], size_node)
    create_child_node(doc, 'height', attrs['height'], size_node)
    create_child_node(doc, 'depth', attrs['depth'], size_node)
    root_node.appendChild(size_node)

    create_child_node(doc, 'segmented', attrs['segmented'], root_node)

    for box in attrs['boxes']:
        object_node = create_object_node(doc, attrs, box)
        root_node.appendChild(object_node)

    write_xml_without_head(doc, anno_file)


def write_xml_without_head(doc, file):

    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent=' ' * 4, newl='\n', encoding='utf-8')
    tmpfile.close()

    fin = open('tmp.xml')
    fout = open(file, 'w')
    lines = fin.readlines()

    for line in lines[1:]:
        if line.split():
            fout.writelines(line)

    fin.close()
    fout.close()
def process_annatation(img_path, anno_path):
    image_path = os.getcwd()+'/data/VOCdevkit2007/VOC2007/JPEGImages'
    image_file_list = os.listdir(img_path)

    for img in image_file_list:
        if os.path.exists(image_path):
            shutil.copy(os.path.join(img_path,img),image_path)
        else:
            os.makedirs(image_path)
            shutil.copy(os.path.join(img_path,img),image_path)
    
    txt_path = anno_path
    xml_path = os.getcwd()+'/data/VOCdevkit2007/VOC2007/Annotations'
    for root, dirs, files in os.walk(txt_path):
        for f in files:
            attrs = {}
            temp = f.replace('.txt','')
            anno_file = xml_path + '/' + temp + '.xml'
            attrs['image_name'] = temp + '.jpg'
            img = image_path+'/'+temp+'.jpg'
            img = Image.open(img)
            width, height = img.size
            attrs['width'] = str(width)
            attrs['height'] = str(height)

            attrs['author'] = "sunxichen"
            attrs['root'] = "annotation"
            attrs['folder'] = "VOC2007"
            attrs['annotation'] = "PASCAL VOC2007"
            attrs['segmented'] = "0"
            attrs['difficult'] = "0"
            attrs['truncated'] = "0"
            attrs['pose'] = "Unspecified"
            attrs['database'] = "Battery"
            attrs['depth'] = "3"
        
            boxes = []
            with open(txt_path+'/'+f,'r') as f1:
                readdata = f1.readlines()
                for i in readdata:
                    ls = i.split(' ')
                    cat = ls[1]
                    if cat != '带电芯充电宝' and cat != '不带电芯充电宝':
                        continue
                    xmin = ls[2]
                    if int(xmin) > width:
                        continue
                    if int(xmin) < 0:
                        xmin = str(1)
                    ymin = ls[3]
                    if int(ymin) < 0:
                        ymin = str(1)
                    xmax = ls[4]
                    if int(xmax) > width: 
                        xmax = str(width - 1)
                    ymax = ls[5].replace('\n','')
                    if int(ymax) > height:
                        ymax = str(height - 1)
                    boxes.append((cat,str(int(xmin)-1),str(int(ymin)-1),str(int(xmax)-1),str(int(ymax)-1)))
            attrs['boxes'] = boxes
            create_xml_file(anno_file,attrs)
    
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                       help='whether use large imag scale',
                       action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                       help='whether use multiple GPUs',
                       action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                       help='whether perform class_agnostic bbox regression',
                       action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=5, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=1999, type=int)
  parser.add_argument('--vis', dest='vis',
                       help='visualization mode',
                       action='store_true')
  parser.add_argument('--img_path', dest='img_path',
                      help='Path to Images',
                      default='/home/woaibritneyspears/Image_test', type=str)
  parser.add_argument('--anno_path', dest='anno_path',
                      help='Path to Annotations files directs',
                      default='/home/woaibritneyspears/Anno_test', type=str)
  parser.add_argument('--test_file', dest='test_file_path',
                      help='Path to test file',
                      default='/home/woaibritneyspears/core_coreless_test.txt', type=str)
  args = parser.parse_args()
  return args            

def test():
    args = parse_args()
    test_txt_path = os.getcwd()+'/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    test_file = args.test_file_path
    img_path = args.img_path
    anno_path = args.anno_path

    f = open(test_file,'r')
    f1 = open(test_txt_path,'w')

    for line in f.readlines():
        f1.write(line.replace('.txt',''))
    f.close()
    f1.close()

    process_annatation(img_path,anno_path)
    run(args)

def get_output():
    origin_dir = os.getcwd() + '/data/VOCdevkit2007/results/VOC2007/Main' 
    output_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/predicted_file'
    file_list = os.listdir(origin_dir)

    for fi in file_list:
        if os.path.exists(output_dir):
            shutil.copy(os.path.join(origin_dir,fi),output_dir)
        else:
            os.makedirs(output_dir)
            shutil.copy(os.path.join(origin_dir,fi),output_dir)

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("需要指定第三个参数测试集txt文件(core_coreless_test.txt)")
    #     print("python test.py img_path anno_path core_coreless_test_path")
    #     exit()
    # img_path = sys.argv[1]
    # anno_path = sys.argv[2]
    # test_file = sys.argv[3]
    test()
    print("Writting predicted file")
    get_output()

