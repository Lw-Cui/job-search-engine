import React from "react";
import JobItem from "./JobItem";

const JobList = ({ jobs, onJobSelect, selectedJob }) => {  
  const renderedList = jobs.map(job => (
    <JobItem key={job.id} onJobSelect={onJobSelect} job={job} selectedJob={selectedJob}/>
  ));

  return (
    <div>
      <div className="ui relaxed divided list">{renderedList}</div>
    </div>
  );
};

export default JobList;
