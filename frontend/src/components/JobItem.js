import React from "react";

const JobItem = ({ job, selectedJob, onJobSelect }) => {
  return (
    <div
      className={`ui link card fluid ${
        selectedJob != null && job.description === selectedJob.description ? "blue" : null // need to change
      }`}
      onClick={() => onJobSelect(job)}
      key={job.id}
    >
      <div className="content">
        <i className="right floated big desktop icon"></i>
        <div className="header">{job.title}</div>
        <div className="meta">{job.company}</div>
      </div>
      <div className="content">
        <ol className="ui list">
          <div className="item">
            <i className="marker icon"></i>
            <div className="content">{job.location}</div>
          </div>
          <div className="item">
            <i className="dollar sign icon"></i>
            <div className="content">$40.00-$70.00 /hour</div>
          </div>
          <div className="item">
            <i className="tag icon"></i>
            <div className="content">
              {job.category.charAt(0).toUpperCase() + job.category.slice(1)}
            </div>
          </div>
          <div className="item">
            <i className="linkify icon"></i>
            <div className="content">
              <a href="http://www.semantic-ui.com">abc.com</a>
            </div>
          </div>
        </ol>
      </div>
    </div>
  );
};

export default JobItem;
