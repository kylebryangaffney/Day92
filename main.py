from flask import Flask, render_template, redirect, url_for, flash
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import SubmitField, FileField, IntegerField
from wtforms.validators import DataRequired, NumberRange
import os
from datetime import datetime
from colorextractor import ColorExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = "8BYkEfBA6O6donzWlSihBXox7C0sKR6b"
app.config['UPLOAD_FOLDER'] = 'uploads'
Bootstrap5(app)

class ColorExtractionForm(FlaskForm):
    image_file = FileField(validators=[DataRequired()])
    colors_to_return = IntegerField('The amount of colors to return, must be less than or equal to 8', validators=[DataRequired(), NumberRange(1, 8)])
    submit = SubmitField('Submit')

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

@app.route("/", methods=['GET', 'POST'])
def home():
    form = ColorExtractionForm()
    if form.validate_on_submit():
        file = form.image_file.data
        num_of_colors = form.colors_to_return.data
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Ensure thread safety and context management for image processing
        try:
            color_extractor = ColorExtractor(file_path, num_of_colors)
            color_extractor.load_image()
            color_extractor.preprocess_image()
            color_extractor.find_most_common_colors()
            colors = color_extractor.get_colors()
            plot_url = color_extractor.plot_colors()
            return render_template('results.html', colors=colors, plot_url=plot_url)
        except ValueError as e:
            flash(str(e), 'danger')
        finally:
            # Ensure that the file is deleted after processing
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return render_template("index.html", form=form)

@app.route("/results")
def results():
    return render_template('results.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
