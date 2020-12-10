from PIL import Image
import matplotlib.pyplot as plt

select_image = 'ship2'

image = Image.open('option2pics/'+select_image+'.jpg')

resized = image.resize((32, 32),Image.ANTIALIAS)

print("Orignal size :", image.size)
print("Resized image size:", resized.size)

plt.imshow(image)
plt.show()

plt.imshow(resized)
plt.show()

resized.save('Resized/R_'+select_image+'.jpg')


