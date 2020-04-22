# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    content = request.json
    print(content)
    job = [
        dict(company="Google",
             title="engineer",
             category="engineering",
             location="NY 淄博",
             description="The Amazon Devices team designs and engineers high-profile consumer electronics, "
                         "including the innovative Cloud Cam and Alexa family of products. We have also produced "
                         "groundbreaking devices such as Echo Show and the Echo Look to add Computer Vision to "
                         "Alexa. What will you help us create?What will you help us create?We are looking for a "
                         "passionate, hard-working, and talented Software Development Engineer who has experience "
                         "building world-class mobile Android apps. The person chosen for this position will have "
                         "the opportunity to contribute their creative ideas and energy to our group. You will be "
                         "working with world class Artificial Intelligence and Machine Learning experts, "
                         "distributed cloud systems and next generation camera streaming devices. The development "
                         "will be from the concept stage to the launch stage and ensuring the highest level of "
                         "quality for your deliverables.You will be:· Responsible for the design, development and "
                         "maintenance of a mobile app that enables innovative computer vision experience· Working "
                         "with other team members to investigate design approaches, prototype new technology and "
                         "evaluate technical feasibility· Leading architecture and design of new and current "
                         "systems, from conception to launch· Working in an Agile/Scrum environment to deliver high "
                         "quality software",
             minimum_qualification="· Bachelor’s degree in Computer Science, Computer Engineering or related field· "
                                   "2+ years professional experience in software development building "
                                   "consumer-facing Android apps· 3+ years overall software development experience· "
                                   "2+ years of development and debugging skills using Java· 2+ years of experience "
                                   "performing Computer Science fundamentals in object-oriented design, "
                                   "data structures, algorithm design, problem solving, and complexity analysis",
             preferred_qualification="· Master’s degree in Computer Science, Computer Engineering or related field· "
                                     "Have designed and developed a complete Android app from top to bottom· Have "
                                     "worked on Android companion apps for consumer electronics devices· Experience "
                                     "in multimedia streaming protocols such as WebRTC. Understanding of video and "
                                     "audio codecs.· Web Services and cloud experience as it relates to mobile "
                                     "apps· Development and debugging skills using C++· Knowledge of professional "
                                     "software engineering practices & best practices for the full software "
                                     "development life cycle, including coding standards, code reviews, "
                                     "source control management, build processes, testing, and operationsAmazon is "
                                     "an Equal Opportunity Employer – Minority / Women / Disability / Veteran / "
                                     "Gender Identity / Sexual Orientation "),

        dict(company="Amazon",
             title="SDE",
             category="engineering",
             location="Settle",
             description="Are you passionate about the intersection of cloud computing, mobile applications, "
                         "and BigData analytics? Join the Amazon Pinpoint team to build large-scale real-time "
                         "analytics application for mobile app developers and enterprises to engage their users. Many "
                         "of the world’s top mobile apps use our services to power their applications. Enterprise "
                         "customers use our services to analyze customer interaction, design and execute a "
                         "multi-channel engagement campaign.AWS Pinpoint services are built using the next generation "
                         "cloud technologies. You will enable our customers to use the serverless microservices way "
                         "to create analytics, an approach that lets customers turn business logic and application "
                         "code into scalable, fault-tolerant production systems without requiring every developer to "
                         "become an expert in distributed systems, deployment technologies, and infrastructure "
                         "management. The best part of working with Amazon Pinpoint is the chance to help customers "
                         "solve exciting new challenges and create new app experiences every day that impact millions "
                         "of users. Our customers are innovators, and you will have the chance to work with them to "
                         "understand their challenges and design new offerings. Together, we’ll shape not just our "
                         "own products, but the direction of the industry. Learn more about our business at "
                         "https://aws.amazon.com/pinpoint/.We are looking for senior software developers who long for "
                         "the opportunity to design and build large-scale systems with global impact. You will get a "
                         "unique opportunity to build full-stack applications in the mobile and web spaces that "
                         "involve working with iOS, Android, and React as well as different AWS services including "
                         "Cognito, Lambda, API Gateway, S3, Kinesis, EMR, Redshift, Dynamo, SNS, Athena, "
                         "and QuickSight. We need your help to build a platform that will ingest trillions of "
                         "messages from billions of devices. You will make applications that leverage big data "
                         "technology and machine learning to improve user engagement with mobile apps. Your messaging "
                         "and targeting applications will power the world’s leading social media, gaming, sports, "
                         "educational, and consumer applications. You will work closely with Product Management to "
                         "best address our customers' needs and help shape the product for success by creating "
                         "engaging and dynamic experiences.What does it take to succeed in this role? You need to be "
                         "creative, responsible, and able to dig deep into AWS emerging technologies. You will think "
                         "about business opportunities, operational issues, architectural diagrams, and the customer "
                         "perspective in the course of a single conversation. You have a deep mastery of programming "
                         "languages, distributed system design, and performance. Someone who makes the team both "
                         "productive and fun to work in. Excited to learn from others while bringing your own novel "
                         "capabilities and perspectives. Our team members thrive in a hands-on environment where "
                         "everyone actively participates in product definition, technical architecture review, "
                         "iterative development, code review, and operations.",
             minimum_qualification="· Bachelor's Degree in Computer Science or related field· Expert knowledge of one "
                                   "of the following programming languages: Java, C and C++· 8+ years of hands on "
                                   "experience in software development, including design, implementation, debugging, "
                                   "and support, building scalable system software and/or Services· Deep "
                                   "understanding of distributed systems and web services technology· Strong at "
                                   "applying data structures, algorithms, and object oriented design, "
                                   "to solve challenging problems· Experience working with REST and RPC service "
                                   "patterns and other client/server interaction models· Track record of building and "
                                   "delivering mission critical, 24x7 production software systems· Bachelor’s degree "
                                   "in Computer Science or equivalent",
             preferred_qualification="· Master Degree or PhD in Computer Science, Computer Engineering or related "
                                     "field.· Experience with BigData technology e.g. Hadoop, and Spark· iOS and "
                                     "Android SDK experienceAmazon is an Equal Opportunity-Affirmative Action "
                                     "Employer – Minority / Female / Disability / Veteran / Gender Identity / Sexual "
                                     "Orientation."),

    ]
    return jsonify(job)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
