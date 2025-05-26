# Sample code: generate animation from still images from tcviewer and custom_code_*.py sample outputs
# View in a browser that supports animated png (like Chrome)
from PIL import Image

image_paths1 = ['screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img1.png', 'screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img2.png', 'screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img3.png', 'screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img4.png', 'screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img5.png', 'screenshots/eps_genesis_caribbean-2024-10-25-09-04-26-img6.png']
image_paths2 = ['screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img01.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img02.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img03.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img04.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img05.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img06.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img07.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img08.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img09.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img10.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img11.png', 'screenshots/det_genesis_caribbean-2024-10-25-09-03-03-img12.png']

image_paths2 = ['screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img01.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img02.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img03.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img04.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img05.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img06.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img07.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img08.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img09.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img10.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img11.png', 'screenshots/det_genesis_caribbean-2024-11-12-20-44-42-img12.png']


"""prefix='eps_genesis_caribbean_3days_10_25_00'
images1 = [Image.open(image_path) for image_path in reversed(image_paths1)]
durations1 = [1000]*(len(image_paths1)-1) + [3000]
images1[0].save(f'screenshots/{prefix}.apng', save_all=True, append_images=images1[1:], duration=durations1, loop=0)
"""
prefix='det_genesis_caribbean_3days_11_11_00'
durations2 = [1000]*(len(image_paths2)-1) + [3000]
images2 = [Image.open(image_path) for image_path in reversed(image_paths2)]
images2[0].save(f'screenshots/{prefix}.apng', save_all=True, append_images=images2[1:], duration=durations2, loop=0)
