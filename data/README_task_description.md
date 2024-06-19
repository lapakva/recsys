# Intro

For your next step in the application process at Instaffo we'd like you to do the task given below to be able to further assess your skills and knowledge. Please send a link to a publicly accessible git repository containing your solution to the person managing your application process (e.g. in the chat on instaffo.com).

The model you chose and it's accuracy is not all that counts. Code quality (including project structure), dependencies and environment management, documentation (docstrings, comments, README file, etc.) are **equally** important!

We wish you good luck (and also a lot of fun) with the task! 🍀

# Matching talents and jobs using machine learning

Instaffo is a recruiting platform that makes money by bringing together hiring companies and talents. Companies offer job opportunities and need the right talents to fill their open job positions, talents on the other hand are looking for new job opportunities.

One core component of the Instaffo platform is the search functionality, which e.g. enables talents to only see relevant job opportunities, the most interesting ones ideally being at the top. This involves:

- Filtering out talents/job opportunities that don't _**match**_ (meaning the job opportunity doesn't fit the talents' profile and should therefore not being recommended to them).
- Ranking all results in a meaningful order.

## Task

In this task you'll create a lightweight search & ranking component using Python and machine learning.

First, let's take a look at the provided data (see _**data.json**_), which contains an array of dictionaries, each having the following properties:

- **talent**: profile information of the talent (detailed description below).
- **job**: profile information of the job (detailed description below).
- **label**: `true` if talent and job _**match**_, else `false`.

The **talent** dictionary contains the following properties:

| Field name         |    Type    |                       Description                       |
| :----------------- | :--------: | :-----------------------------------------------------: |
| degree             |    str     |              Highest degree of the talent               |
| job_roles          | list[str]  |          Job roles the talent is interested in          |
| languages          | list[dict] | Languages the talent speaks with their respective level |
| salary_expectation |    int     |  Salary the talent expects a potential new job to pay   |
| seniority          |    str     |                 Seniority of the talent                 |


The **job** dictionary contains the following properties:

| Field name  |    Type    |               Description               |
| :---------- | :--------: | :-------------------------------------: |
| job_roles   | list[str]  |    Applicable job roles for the job     |
| languages   | list[dict] |    Language requirements of the job     |
| max_salary  |    int     |   Maximum salary the job is offering    |
| min_degree  |    str     |  Minimum degree a talent needs to have  |
| seniorities | list[str]  | Seniorities talents are allowed to have |


### Part 1

Create a machine learning model which takes a **talent** and **job** profile as input and returns a label (`true` if talent and job _**match**_, else `false`) and a score (a float, for ranking purposes only).

The goal of this task is **not** to achieve the highest accuracy. The importance is to choose an approach which is appropriate (this also includes data cleaning, transformation and feature extraction). Hence, there is no need to do any hyper parameter tuning or similar. Just use sensible values where applicable (e.g. the defaults).

Important requirements:

- Python is the only programming language allowed.
- No external services (e.g. APIs like ChatGPT), databases (e.g. MongoDB) or search/recommender systems (e.g. Elasticsearch) are allowed.
- Only `scikit-learn` is allowed as machine learning library/framework.
- Besides the restriction regarding machine learning libraries, every publicly available Python package can be used.

Your solution for this task should include the following:

- All code (Python scripts, modules, jupyter notebooks, etc.) required for training the machine learning model, including data cleaning, data transformation and feature extraction.
- A **short** (5 sentences are already enough) description explaining why you chose the approach above.
- If the model size is too big you don't need to push it into git.

**⚠️ The machine learning model and potentially some functions you create here will be used in the second part of the task. It's wise to read it before starting with this task to already create reusable code for the second part.**

### Part 2

Write a search & ranking component using the provided template (see _**search.py**_) with the machine learning model you created in the first part of this task.

This part is not about being able to accomplish the task (we expect each candidate to easily being able to implement everything necessary) but showing us that you have the necessary software development skills required for the position. Here it's all about code quality! The provided template only defines one module with methods you **must** implement, not meaning that using additional modules/functions/methods (e.g. for better structuring the code) isn't allowed. We encourage you to use all functionality Python offers to create a proper project!

Important requirements:

- Same requirements as above apply.

Your solution for this task should include the following:

- A complete & proper Python project with everything that's necessary and considered best practice.
- **No** tests or CI/CD is required.