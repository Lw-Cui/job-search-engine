import React from "react";
import SearchBar from "./SearchBar";
import JobList from "./JobList";
import JobDetail from "./JobDetail";
import action from "../apis/action";

class App extends React.Component {
  state = { jobs: [], selectedJob: null };

  // https://stub-dot-en-601-666.ue.r.appspot.com/
  componentDidMount() {
    this.onTermSubmit("");
  }

  onJobSelect = job => {
    this.setState({ selectedJob: job });
  };

  onTermSubmit = async (term, way) => {
    // console.log(term);
    // console.log(way);
    const response = await action.get(term, {
      params: {
        query: term
      }
    });
    // console.log(response);
    this.setState({
      jobs: response.data,
      selectedJob: response.data[0]
    });
  };

  render() {
    return (
      <div className="ui container" style={{ marginTop: "10px" }}>
        <SearchBar onFormSubmit={this.onTermSubmit} />
        <div className="ui grid">
          <div className="ui row">
            <div className="six wide column">
              <JobList
                onJobSelect={this.onJobSelect}
                jobs={this.state.jobs}
                selectedJob={this.state.selectedJob}
              />
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
