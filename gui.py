import tkinter
from PIL import ImageGrab, Image

root = tkinter.Tk()

canvas = tkinter.Canvas(root, width = 400, height = 400, bg = '#000')

def getter():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    
    image = ImageGrab.grab().crop(tuple(map(lambda x: x * 2, (x, y, x1, y1))))
    image.save('test.png')
    image.thumbnail((10, 10), Image.ANTIALIAS)
    image.save('test_rsz.png')
    print(text.get())

button = tkinter.Button(root, text = 'Submit', command = getter)
text = tkinter.Entry(root)




def paint(event):
   x1, y1 = ( event.x - 10 ), ( event.y - 10 )
   x2, y2 = ( event.x + 10 ), ( event.y + 10 )
   canvas.create_oval( x1, y1, x2, y2, fill = '#fff', outline = '#fff')

def keyDown(event):
	print('push')

canvas.bind('<B1-Motion>', paint)
canvas.bind('<Return>', getter)

button.bind()

canvas.pack()
button.pack()
text.pack()


root.mainloop()
