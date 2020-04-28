import React from "react";

class SearchBar extends React.Component {
  state = { term: "" };

  onFormSubmit = event => {
    event.preventDefault();
    this.props.onFormSubmit(this.state.term);
  };

  render() {
    return (
      <div className="ui segment">
        <form className="ui form" onSubmit={this.onFormSubmit}>
          <div className="ui fluid icon input">
            <input
              type="text"
              placeholder="Search..."
              value={this.state.term}
              onChange={e => this.setState({ term: e.target.value })}
            />
            <i className="search icon"></i>
          </div>
        </form>
      </div>
    );
  }
}

export default SearchBar;
