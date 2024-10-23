# Sample code: generate animation from still images from tcviewer and custom_code_*.py sample outputs
# View in a browser that supports animated png (like Chrome)
from PIL import Image

image_paths1 = ['screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img1.png', 'screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img2.png', 'screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img3.png', 'screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img4.png', 'screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img5.png', 'screenshots/eps_genesis_caribbean-2024-10-23-03-05-39-img6.png']
image_paths2 = ['screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img01.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img02.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img03.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img04.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img05.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img06.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img07.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img08.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img09.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img10.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img11.png', 'screenshots/det_genesis_caribbean-2024-10-23-03-27-05-img12.png']

prefix='eps_genesis_caribbean_3days'
images1 = [Image.open(image_path) for image_path in reversed(image_paths1)]
durations1 = [1000]*(len(image_paths1)-1) + [3000]
images1[0].save(f'screenshots/{prefix}.apng', save_all=True, append_images=images1[1:], duration=durations1, loop=0)

prefix='det_genesis_caribbean_3days'
durations2 = [1000]*(len(image_paths2)-1) + [3000]
images2 = [Image.open(image_path) for image_path in reversed(image_paths2)]
images2[0].save(f'screenshots/{prefix}.apng', save_all=True, append_images=images2[1:], duration=durations2, loop=0)
