import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import subprocess
import os
import cv2
import numpy as np
from PIL import Image, ImageTk


class InpaintingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("인페인팅 컨트롤 애플리케이션")
        self.root.geometry("500x500")
        self.root.configure(bg='#f0f0f5')
        # Frame for buttons
        self.button_frame = tk.Frame(root, bg='#f0f0f5')
        self.button_frame.pack(pady=20)

        # Buttons for operations
        self.org_button = tk.Button(self.button_frame, text="Org모델 Inpainting하기", width=20,
                                    command=self.run_org_inpainting)
        self.org_button.grid(row=0, column=0, padx=10)

        self.prop_button = tk.Button(self.button_frame, text="Prop모델 Inpainting하기", width=20,
                                     command=self.run_prop_inpainting)
        self.prop_button.grid(row=0, column=1, padx=10)

        self.display_button = tk.Button(self.button_frame, text="결과 확인", width=20,
                                        command=self.display_results)
        self.display_button.grid(row=1, column=0, padx=10, pady=10)

        self.quit_button = tk.Button(self.button_frame, text="종료", width=20, command=self.root.quit)
        self.quit_button.grid(row=1, column=1, padx=10, pady=10)

        # Label for instructions
        self.instruction_label = tk.Label(root, text="인페인팅을 실행하고 결과를 확인해보자", font=("Arial", 12))
        self.instruction_label.pack(pady=10)

        # Canvas for image display
        self.canvas = tk.Canvas(root, width=400, height=300, bg='gray')
        self.canvas.pack(pady=20)

        #self.canvas.create_text(400,400,text="인페인팅 실험하기", font=("Arial",16),fill="white")


        self.initial_image_path='../FINAL_TEST/Image_Path/image1.png'
        self.show_image_on_canvas(self.initial_image_path)


    def run_org_inpainting(self):
        print("Running org_inpainting.py...")
        subprocess.run(["python", "org_inpainting.py"], cwd=os.path.join("..", "DeepFillv2-TF2-org"))
        messagebox.showinfo("Info", "Org inpainting process completed.")

    def run_prop_inpainting(self):
        print("Running prop_inpainting.py...")
        subprocess.run(["python", "prop_inpainting.py"], cwd=os.getcwd())
        messagebox.showinfo("Info", "Prop inpainting process completed.")

    def display_results(self):
        result_dir = "../FINAL_TEST/Result_Path"
        org_result_path = os.path.join(result_dir, "Org_Result.png")
        prop_result_path = os.path.join(result_dir, "Prop_Result.png")

        org_image = cv2.imread(org_result_path)
        prop_image = cv2.imread(prop_result_path)

        if org_image is None:
            print("원본 이미지 결과가 없습니다.")
        elif prop_image is None:
            print("개선안 이미지 결과가 없습니다.")

        else:
            cv2.imshow("Org_Result", org_image)
            cv2.imshow("Prop_Result", prop_image)

            print("Prees q to close")

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
        messagebox.showinfo("Info", "Showing result Org and Prop completed.")

    def show_image_on_canvas(self, image_path_or_array):
        # Convert image to PIL format and then to ImageTk format

        if isinstance(image_path_or_array,str):
            image=cv2.imread(image_path_or_array)
        else:
            image=image_path_or_array

        width, height=300,200
        image=cv2.resize(image,(width,height))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(pil_image)

        # Clear the canvas and display the new image
        self.canvas.delete("all")
        self.canvas.create_image(200, 150, image=photo)
        self.canvas.image = photo  # Keep a reference to avoid garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintingApp(root)
    root.mainloop()
