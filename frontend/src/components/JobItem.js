import React from "react";

const JobItem = ({ job, onJobSelect }) => {

  return (
    <div
      className={"ui link card fluid"}
      onClick={() => onJobSelect(job)}
      key={job.id}
    >
      <div className="content">
        <i className="right floated large desktop icon"></i>
        <div className="header">{job.title}</div>
        <div className="meta">Google</div>
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
            <i className="mail icon"></i>
            <div className="content">
              <a href="mailto:jack@semantic-ui.com">abc.com</a>
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
