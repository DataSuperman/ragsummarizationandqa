# Task summary and technical details

When you run the application you will receive an interactive session in shell where 
you can interact with the tasks. Alternatively there is a jupyter notebook `Task summary and technical details.ipynb` that explains  
various parts of the code into digestable chunks. 
## Running the application

Using docker:
```bash
docker build -t src .
docker run -it --rm src

```

Or alternatively, you can run the application using  `make`:
```bash
make docker_run
```

On linux you can run locally using:
```bash
make run
```

## Sample output
Summaraization of the dataset:
```

**************************************************
Type s for summary, q for question answering system, x if you want to quit 
s
.
**************************************************
```

Question answering system:
```
**************************************************
Type s for summary, q for question answering system, x if you want to quit 
q
--------------------------------------------------
Question (or nothing to quit): What are the next steps?
Answer is  grounded in the facts
Answer: The next steps involve setting up a meeting to discuss pricing on Monday at 4:30 pm. The focus will be on implementing routines for coaching and improving sales techniques. Workshops and pitch tests are currently being used to enhance skills and performance.
--------------------------------------------------
Question (or nothing to quit):  Who is Lancelot ?    
Answer is  grounded in the facts
Answer: Lancelot is the CEO of SquareChair, which is in the same incubator as the speaker. The speaker has not specifically talked to Lancelot about certain topics. They plan to discuss with Lancelot the following week.
--------------------------------------------------
