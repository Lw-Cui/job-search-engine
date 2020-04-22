import React from "react";
import SearchBar from "./SearchBar";
import JobList from "./JobList";
import JobDetail from "./JobDetail";

class App extends React.Component {
  temp = [
    {
      id:1,
      title: "Software Engineer",
      location: "San Francisco, CA",
      responsibilities: [
        "Be an active and integral member of your team",
        "Channel your enthusiasm to innovate",
        "Be part of a supportive team that develops and deploys technologies that matter to our national security",
        "Work with agility, flexibility, and collaboration to explore creative ideas"
      ],
      basicQualifications: [
        "Bachelor's degree with 0-2 years of related experience or 4 years of related experience in lieu of a degree",
        "Experience in higher level programming languages",
        "Demonstrated experience quickly learning new concepts and approaches",
        "Must have a US Citizenship with current or active TS/SCI Access"
      ],
      preferredQualifications: [
        "Languages such as C++, Python",
        "Technology and platforms like Linux Red Hat",
        "Computer security and regulations",
        "Desire to work across teams",
        "Desire to find and resolve problems in Software, as well as develop from a clean sheet"
      ]
    },
    {
      id:2,
      title: "Software Developer",
      location: "New York, NY",
      responsibilities: [
        "...",
        "...",
      ],
      basicQualifications: [
        "...",
        "...",
        "...",
        "...",
      ],
      preferredQualifications: [
        "...",
        "...",
        "...",
        "...",
      ]
    }
  ];

  state = { jobs: this.temp, selectedJob: null };

  componentDidMount() {}

  onJobSelect = job => {
    this.setState({ selectedJob: job });
  };


  onTermSubmit() {
    // need to do
  }

  render() {
    return (
      <div className="ui container" style={{ marginTop: "10px" }}>
        <SearchBar onFormSubmit={this.onTermSubmit} />
        <div className="ui grid">
          <div className="ui row">
            <div className="six wide column">
              <JobList onJobSelect={this.onJobSelect} jobs={this.state.jobs} />
            </div>
            <div className="ten wide column">
              <JobDetail job={this.state.selectedJob} />
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
