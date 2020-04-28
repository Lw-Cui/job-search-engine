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
  // const renderedRes = job.responsibilities.map(res => <li>{res}</li>);
  // const renderedbQ = job.minimum_qualification.map(bq => <li>{bq}</li>);
  // const renderedpQ = job.preferred_qualification.map(pq => <li>{pq}</li>);

  const renderedList = (
    <div>
      <div className="content">
        <h4 className="ui header">What does success look like?</h4>
        As a successful employee, you will:
        <ul className="ui list">{job.description}</ul>
      </div>
      <br />
      <div className="content">
        <h4 className="ui header">Basic Qualifications: </h4>
        <ul className="ui list">{job.minimum_qualification}</ul>
      </div>
      <br />
      <div className="content">
        <h4 className="ui header">Preferred Qualifications: </h4>
        Proficiency and interest in some of the below is highly beneficial for
        this role, and we will teach you what you don't know:
        <ul className="ui list">{job.minimum_qualification}</ul>
      </div>
    </div>
  );

  return <div>{renderedList}</div>;
};

export default JobDetail;
