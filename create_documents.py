import csv
import pickle
from tqdm import tqdm
from termcolor import cprint 
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics

from utils import get_args, make_dir
from variables import * 


cprint('\n\nGENERATING PDF FILES FOR :', 'yellow', attrs=['bold'])
cprint(f'-DATASET : {place_name}', 'green')
cprint(f'-SAVING AS : {save_as}\n', 'green')


data = f"train_data/{place_name}_data"

cprint('Loading data...', 'light_yellow')

f = open(data, 'rb')
dataset = pickle.load(f)
f.close()

cprint('Dataset loaded successfully!\n', 'light_green')

cprint('Generating documents...', 'light_yellow')

if args.save_as == 'one_path_one_doc':

    cprint("Unsupported right now, please use the 'save_as' argument with 'all_paths_one_doc'.", 'red')
       

# Save all the textual representations of each path in one big document
elif args.save_as == 'all_paths_one_doc':
		
    make_dir('pdf_docs')
    my_doc = SimpleDocTemplate(f'pdf_docs/{place_name}_paths.pdf')

    sample_style_sheet = getSampleStyleSheet()

    pdfmetrics.registerFont( 
        TTFont('dengxian', 'Dengl.ttf') 
    ) 

    custom_body_style = sample_style_sheet['BodyText']
    custom_body_style.fontName = 'dengxian'
    custom_body_style.fontSize = 10
    custom_body_style.spaceAfter = 15

    flowables = []

    for path_info in tqdm(dataset, dynamic_ncols=True):
        original_path_road_names = path_info['original_path_road_names']
        shortest_path_road_names = path_info['shortest_path_road_names']
        fastest_path_road_names = path_info['fastest_path_road_names']

        starting_address = original_path_road_names[0]
        destination_address = original_path_road_names[-1]

        text = f"从{starting_address}到{destination_address}的最常走的路线是经过{','.join(original_path_road_names)}这几个路段, \
        但从{starting_address}到{destination_address}的最短的路线是经过{','.join(shortest_path_road_names)}这些路，\
        另外同样从{starting_address}到{destination_address}的最快路线是经过{','.join(fastest_path_road_names)}这些路。"
        paragraph = Paragraph(text, custom_body_style)
        flowables.append(paragraph)
        
    my_doc.build(flowables)
    