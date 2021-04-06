from server.app import app

model_path = './training/model/regressor.pkl'
  
if __name__ == "__main__":
        app.run(debug=False)