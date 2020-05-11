import React from "react";

const JobDetail = ({ job }) => {
  if (!job) {
    return (
      <div className="ui placeholder segment">
        <div className="ui icon header">
          <i className="search icon"></i>
          Click the job for details
        </div>
      </div>
    );
  }

  const renderedList = (
    <div>
      <div className="content">
        <h4 className="ui header">Description</h4>
        <ul className="ui list">{job.description}</ul>
      </div>
      <br />
      <div className="content">
        <h4 className="ui header">Minimum Qualifications: </h4>
        <ul className="ui list">{job.minimum_qualifications}</ul>
      </div>
      <br />
      <div className="content">
        <h4 className="ui header">Preferred Qualifications: </h4>
        <ul className="ui list">{job.preferred_qualifications}</ul>
      </div>
    </div>
  );

  return <div>{renderedList}</div>;
};

export default JobDetail;
